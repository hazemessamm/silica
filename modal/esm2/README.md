# Setup

## Install Modal client
```pip install --upgrade modal-client```

## Authenticate Modal client
```modal token new```

## Add BigQuery Service Account to Modal Secrets
* On the Modal web console, open [Secrets page](https://modal.com/secrets/)
* Select "Create new secret" button
* Select "Google Cloud" template button
* Paste `[get key from an admin]` into the "Value" box and select "Next".
* Name the key `gcp-biolm-hackathon-bq-secret` and select "Create"

## Run the application (no writes to BQ)
```python app.py```

## Run the application and write results to BQ
* Modify the `write_to_bq = False` line, setting it to `True`.
* Modify the `username = "example-user"` line, setting it to unique identifier for you
* Run the applicaition ```python app.py``` nd use the ```example_screen_sequences_metadata``` for fine grained metadata control
* Check to see that your results are written to the [BQ web console](https://console.cloud.google.com/bigquery?referrer=search&authuser=0&orgonly=true&project=biolm-hackathon&supportedpurview=organizationId&ws=!1m10!1m4!1m3!1sbiolm-hackathon!2sbquxjob_7c15a94f_192845bbde8!3sUS!1m4!4m3!1sbiolm-hackathon!2sscreened_sequences!3sesm2_screen) to the `biolm-hackathon.screened_sequences.esm2_screen` (default table) in the `biolm-hackathon` project
