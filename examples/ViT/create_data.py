import os
import io
import lancedb
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["AWS_ACCESS_KEY_ID"] = "YOUR_AWS_ACCESS_KEY_ID                                                                                                                                           "
os.environ["AWS_SECRET_ACCESS_KEY"] = "YOUR_AWS_SECRET_ACCESS_KEY                                                                                                                   "
os.environ["AWS_REGION"] = "us-east-2"
LANCEDB_API_KEY = "YOUR_LANCE_API_KEY                                                                                                                                           "

NUM_IMAGES = 10000
NUM_CLASSES = 10
IMAGE_SHAPE = (392, 392, 3)
TBL_NAME = f"images_{IMAGE_SHAPE[0]}"
BASE_DIR = "./benchmark_data"
IMAGE_DIR = os.path.join(BASE_DIR, "pytorch_image_folder", TBL_NAME)
PARQUET_FILE_PATH = os.path.join(BASE_DIR, f"{TBL_NAME}.parquet")

S3_BUCKET_NAME = "lancedb-datasets-dev-us-east-2-devrel"
S3_BASE_PREFIX = f"training/mfu_test_{IMAGE_SHAPE[0]}"
S3_PARQUET_KEY = f"training/mfu_test_parquet/{TBL_NAME}.parquet"

def generate_datasets():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    for i in range(NUM_CLASSES):
        os.makedirs(os.path.join(IMAGE_DIR, str(i)), exist_ok=True)

    print("Connecting to LanceDB Enterprise...")
    db = lancedb.connect(
        uri="db://training",
        api_key=LANCEDB_API_KEY,
        host_override="https://ayush:YOUR_PASSWORD  @devrelaws@3-129-160-110.sslip.io",
        region=os.getenv("LANCEDB_REGION", "us-east-1"),
    )

    if TBL_NAME in db.table_names():
        db.drop_table(TBL_NAME)

    print(f"\nPhase 1: Generating {NUM_IMAGES} images...")
    print(" -> Saving locally, pushing to LanceDB, and writing to Parquet...")

    lance_data = []

    schema = pa.schema([
        pa.field("image_bytes", pa.binary()),
        pa.field("label", pa.int32())
    ])
    pq_writer = pq.ParquetWriter(PARQUET_FILE_PATH, schema)

    for i in range(NUM_IMAGES):
        img_arr = np.random.randint(0, 256, IMAGE_SHAPE, dtype=np.uint8)
        label = i % NUM_CLASSES
        img = Image.fromarray(img_arr)

        # Save locally for the S3 raw JPEG upload
        img_path = os.path.join(IMAGE_DIR, str(label), f"img_{i}.jpg")
        img.save(img_path, format="JPEG")

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        lance_data.append({
            "image_bytes": buffer.getvalue(),
            "label": label
        })

        # Batch insert every 2500 images
        if len(lance_data) >= 2500:
            # A. Push to LanceDB
            if TBL_NAME not in db.table_names():
                db.create_table(TBL_NAME, data=lance_data, schema=schema)
            else:
                db.open_table(TBL_NAME).add(lance_data)

            # B. Write row group to Parquet
            batch_table = pa.Table.from_pylist(lance_data, schema=schema)
            pq_writer.write_table(batch_table)

            print(f"  -> Processed batch: {i + 1}/{NUM_IMAGES} complete...")
            lance_data = []

    pq_writer.close()
    print("\nPhase 1 Complete! Data populated across Disk, LanceDB, and Parquet.")

def sync_to_s3():
    print("\nPhase 2: Syncing data to S3...")
    s3_client = boto3.client('s3', region_name=os.getenv("AWS_REGION", "us-east-2"))

    # 1. Upload the monolithic Parquet file
    print(f" -> Uploading {PARQUET_FILE_PATH} to S3...")
    s3_client.upload_file(PARQUET_FILE_PATH, S3_BUCKET_NAME, S3_PARQUET_KEY)
    print(" -> Parquet upload complete!")

    # 2. Gather all local JPEG paths
    print(" -> Gathering raw JPEGs for upload...")
    files_to_upload = []
    for root, _, files in os.walk(IMAGE_DIR):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, IMAGE_DIR)
            s3_key = f"{S3_BASE_PREFIX}/{relative_path}".replace("\\", "/")
            files_to_upload.append((local_path, s3_key))

    uploaded_count = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(s3_client.upload_file, local, S3_BUCKET_NAME, s3_k): s3_k
            for local, s3_k in files_to_upload
        }

        for future in as_completed(futures):
            uploaded_count += 1
            if uploaded_count % 1000 == 0:
                print(f"  -> Uploaded {uploaded_count}/{len(files_to_upload)} raw JPEGs to S3...")

    print("\nAll datasets successfully staged in S3 and LanceDB!")

if __name__ == "__main__":
    generate_datasets()
    sync_to_s3()
