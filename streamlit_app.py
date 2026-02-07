import asyncio  # Lets us call async functions from normal (sync) Streamlit code.
from pathlib import Path  # Cross-platform path handling (avoids manual string paths).
import time  # Used for sleeps and polling timeouts.

import streamlit as st  # Streamlit UI framework (widgets + app state + rendering).
import inngest  # Inngest client SDK: lets us send events to Inngest.
from dotenv import load_dotenv  # Loads values from a local .env file into environment variables.
import os  # Access environment variables like INNGEST_API_BASE.
import requests  # Simple HTTP client to call the local Inngest API for run output.

load_dotenv()  # Read .env and set process env vars (e.g., INNGEST_API_BASE).

st.set_page_config(
    page_title="RAG Ingest PDF",  # Browser tab title.
    page_icon="📄",  # Tab icon.
    layout="centered",  # Center the content (instead of wide layout).
)


@st.cache_resource  # Cache this object for the life of the Streamlit process.
def get_inngest_client() -> inngest.Inngest:
    # Create an Inngest client used to send events.
    # app_id must match what your server-side Inngest functions expect.
    # is_production=False makes it behave like local/dev (no prod assumptions).
    return inngest.Inngest(app_id="rag_app", is_production=False)


def save_uploaded_pdf(file) -> Path:
    # Streamlit's uploader returns an UploadedFile-like object.
    # We persist it to disk so the backend can read it by path.

    uploads_dir = Path("uploads")  # Local folder where uploaded PDFs will be stored.
    uploads_dir.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists.

    file_path = uploads_dir / file.name  # Target path for the uploaded PDF.
    file_bytes = file.getbuffer()  # Efficiently get the uploaded bytes.
    file_path.write_bytes(file_bytes)  # Write bytes to disk.

    return file_path  # Return a Path object for downstream use.


async def send_rag_ingest_event(pdf_path: Path) -> None:
    # Send an Inngest event that triggers the "Rag: Ingest PDF" function.
    # This is async because the Inngest client uses async I/O.

    client = get_inngest_client()  # Reuse cached client.
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",  # Must match the trigger name in main.py.
            data={
                # Backend reads this absolute path to load the PDF file.
                "pdf_path": str(pdf_path.resolve()),
                # A stable ID for where these chunks came from (e.g., file name).
                "source_id": pdf_path.name,
            },
        )
    )


st.title("Upload a PDF to Ingest")  # Big header.
uploaded = st.file_uploader(
    "Choose a PDF",  # Widget label.
    type=["pdf"],  # Only allow PDFs.
    accept_multiple_files=False,  # Single file upload.
)

if uploaded is not None:  # Only run this block when the user has selected a file.
    with st.spinner("Uploading and triggering ingestion..."):
        path = save_uploaded_pdf(uploaded)  # Save the upload so backend can read it.

        # Streamlit code runs synchronously, so we use asyncio.run(...) to execute
        # our async event-sending function and wait until the send completes.
        asyncio.run(send_rag_ingest_event(path))

        # Small pause purely for UX (gives user time to see the spinner).
        time.sleep(0.3)

    st.success(f"Triggered ingestion for: {path.name}")  # Show confirmation.
    st.caption("You can upload another PDF if you like.")  # Small helper text.

st.divider()  # Horizontal divider between ingest section and query section.
st.title("Ask a question about your PDFs")  # Header for the Q&A section.


async def send_rag_query_event(question: str, top_k: int) -> None:
    # Send an Inngest event that triggers the "RAG: Search PDF" function.
    # We return the event ID (Inngest uses it to locate the resulting runs).

    client = get_inngest_client()  # Reuse cached client.
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",  # Must match the trigger name in main.py.
            data={
                "question": question,  # The user's query.
                "top_k": top_k,  # How many matching chunks to retrieve.
            },
        )
    )

    # Inngest returns a list of event IDs (because send can accept multiple events).
    return result[0]


def _inngest_api_base() -> str:
    # Base URL for the local Inngest dev API.
    # You can override it by setting INNGEST_API_BASE in your environment/.env.
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    # Call the Inngest HTTP API to list runs created for a given event.
    url = f"{_inngest_api_base()}/events/{event_id}/runs"  # Endpoint for event runs.

    resp = requests.get(url)  # Make the GET request.
    resp.raise_for_status()  # Raise an exception for non-2xx responses.

    data = resp.json()  # Parse JSON response body.
    return data.get("data", [])  # Runs are typically under the "data" key.


def wait_for_run_output(
    event_id: str,
    timeout_s: float = 120.0,
    poll_interval_s: float = 0.5,
) -> dict:
    # Poll the Inngest API until the run completes, then return its output.
    # This turns an async workflow into something the Streamlit UI can await
    # via a blocking loop.

    start = time.time()  # Record start time for timeout handling.
    last_status = None  # Keep the last seen status for better error messages.

    while True:  # Poll until we return output or hit error/timeout.
        runs = fetch_runs(event_id)  # Ask Inngest which runs exist for this event.
        if runs:
            run = runs[0]  # Use the most recent/first run.
            status = run.get("status")  # Run status (string).
            last_status = status or last_status  # Remember status if present.

            # Consider several common "done" statuses.
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}  # Output is what the function returned.

            # Treat hard failures as exceptions.
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")

        # Enforce a maximum wait time to avoid hanging forever.
        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for run output (last status: {last_status})"
            )

        time.sleep(poll_interval_s)  # Sleep before polling again.


with st.form("rag_query_form"):
    # Forms in Streamlit batch widget interactions and only run when submitted.

    question = st.text_input("Your question")  # Text box for the user query.
    top_k = st.number_input(
        "How many chunks to retrieve",  # Label.
        min_value=1,  # Lower bound.
        max_value=20,  # Upper bound.
        value=5,  # Default.
        step=1,  # Increment.
    )
    submitted = st.form_submit_button("Ask")  # Button that submits the form.

    if submitted and question.strip():  # Only proceed if user clicked and question isn't empty.
        with st.spinner("Sending event and generating answer..."):
            # Send the query event to Inngest (async function executed synchronously).
            event_id = asyncio.run(send_rag_query_event(question.strip(), int(top_k)))

            # Poll the Inngest API until the function run completes.
            output = wait_for_run_output(event_id)

            # Extract fields from the function's output payload.
            answer = output.get("answer", "")
            sources = output.get("sources", [])

        st.subheader("Answer")  # Section header.
        st.write(answer or "(No answer)")  # Render the answer text.

        if sources:  # Only show sources section if any were returned.
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")  # Render each source as a bullet line.
