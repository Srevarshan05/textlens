from textlens.batch import BatchOCR


input_folder = r"C:\Users\Srevarshan\OneDrive\Desktop\Projects\textlens\Check_batch_ocr"

batch = BatchOCR(
    model="glm-ocr",
    workers=1,
    output_format="json",
    output_dir="./batch_output",
    enable_dashboard=True,
)

results = batch.run(input_folder)

print(f"Processed: {sum(task.status.value == 'COMPLETED' for task in results)}")
print(f"Failed: {sum(task.status.value == 'FAILED' for task in results)}")
print("Results: ./batch_output")
