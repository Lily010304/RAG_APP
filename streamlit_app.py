import asyncio  # Lets us call async functions from normal (sync) Streamlit code.
from pathlib import Path  # Cross-platform path handling (avoids manual string paths).
import time  # Used for sleeps and polling timeouts.

import streamlit as st  # Streamlit UI framework (widgets + app state + rendering).
import inngest  # Inngest client SDK: lets us send events to Inngest.
from dotenv import load_dotenv  # Loads values from a local .env file into environment variables.
import os  # Access environment variables like INNGEST_API_BASE.
import requests  # Simple HTTP client to call the local Inngest API for run output.
import io # handeling file bytes in memory without saving to disk
from supabase import create_client, Client # Supabase client for cloud storage

load_dotenv()  # Read .env and set process env vars (e.g., INNGEST_API_BASE).

st.set_page_config(
    page_title="RAG Ingest PDF",  # Browser tab title.
    page_icon="📄",  # Tab icon.
    layout="centered",  # Center the content (instead of wide layout).
)

@st.cache_resource
def get_supabase_client() -> Client:
    # create supabase client for uploading PDF files
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )

@st.cache_resource  # Cache this object for the life of the Streamlit process.
def get_inngest_client() -> inngest.Inngest:
    # Create an Inngest client used to send events.
    # app_id must match what your server-side Inngest functions expect.
    return inngest.Inngest(app_id="rag_app",
                           # This key is read from Streamlit's secrets (not .env on Cloud)
        event_key=st.secrets["INNGEST_API_KEY"],
        # This tells the client to send events to Inngest Cloud
        api_base_url="https://api.inngest.com", is_production=True)


def save_pdf_to_supabase(file) -> str:
    # Upload PDF to supabase cloud and return the public url.

    client = get_supabase_client()


    # Generate a unique filename to avoid collisions
    timestamp = int(time.time())
    unique_f_name = f"{timestamp}_{file.name}"

    # get filebytes from file uploader from streamlit
    file_bytes = file.getbuffer().tobytes()

    try:
        result = client.storage.from_("pdfs").upload(
        path=unique_f_name,
        file=file_bytes,
        file_options={"content-type": "application/pdf"}
    )

        st.write(result)

    except Exception as e:
        st.error(str(e))
        raise

    # Get the public url for the uploaded pdf
    public_url = client.storage.from_("pdfs").get_public_url(unique_f_name)
    return public_url

    

async def send_rag_ingest_event(pdf_url: str, source_id: str) -> None:
    # Send an Inngest event that triggers the "Rag: Ingest PDF" function.
    # This is async because the Inngest client uses async I/O.

    client = get_inngest_client()  # Reuse cached client.
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",  # Must match the trigger name in main.py.
            data={
                # Backend reads this absolute path to load the PDF file.
                "pdf_url": pdf_url,
                # A stable ID for where these chunks came from (e.g., file name).
                "source_id": source_id,
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
        # Upload to a supabase storage and get public url
        pdf_url = save_pdf_to_supabase(uploaded)


        # Streamlit code runs synchronously, so we use asyncio.run(...) to execute
        # our async event-sending function and wait until the send completes.
        asyncio.run(send_rag_ingest_event(pdf_url, uploaded.name))

        # Small pause purely for UX (gives user time to see the spinner).
        time.sleep(0.3)

    st.success(f"Triggered ingestion for: {uploaded.name}")  # Show confirmation.
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
    # On Streamlit Cloud, use st.secrets instead of os.getenv.
    return st.secrets.get("INNGEST_API_BASE") or os.getenv("INNGEST_API_BASE")


def fetch_runs(event_id: str) -> list[dict]:
    # Call the Inngest HTTP API to list runs created for a given event.
    url = f"{_inngest_api_base()}/events/{event_id}/runs"  # Endpoint for event runs.
    # since we moved to cloud we need to provide it with authorization aka api key for inngest
    headers = {"Authorization": f"Bearer {st.secrets['INNGEST_SIGNING_KEY']}"}
    resp = requests.get(url, headers=headers)  # Make the GET request.
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
