import { useRef, useState } from "react";

const MAX_FILE_SIZE = 10 * 1024 * 1024;

const ALLOWED_TYPES = [
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/bmp",
];

function ImageUploader({
  onFileSelected,
  onAnalyze,
  loading,
}) {
  const inputRef = useRef(null);

  const [file, setFile] = useState(null);
  const [error, setError] = useState("");
  const [dragActive, setDragActive] = useState(false);

  function validateFile(selectedFile) {
    if (!selectedFile) {
      return "Please select an image.";
    }

    if (!ALLOWED_TYPES.includes(selectedFile.type)) {
      return "Unsupported file type. Use JPG, PNG, WEBP, or BMP.";
    }

    if (selectedFile.size > MAX_FILE_SIZE) {
      return "Image size must be less than 10 MB.";
    }

    return null;
  }

  function processFile(selectedFile) {
    const validationError = validateFile(selectedFile);

    if (validationError) {
      setError(validationError);
      setFile(null);
      return;
    }

    setError("");
    setFile(selectedFile);
    onFileSelected?.(selectedFile);
  }

  function handleInputChange(event) {
    const selectedFile = event.target.files?.[0];

    processFile(selectedFile);
  }

  function handleDrop(event) {
    event.preventDefault();

    setDragActive(false);

    const droppedFile = event.dataTransfer.files?.[0];

    processFile(droppedFile);
  }

  function handleDragOver(event) {
    event.preventDefault();
    setDragActive(true);
  }

  function handleDragLeave(event) {
    event.preventDefault();
    setDragActive(false);
  }

  function openFilePicker() {
    inputRef.current?.click();
  }

  function clearFile() {
    setFile(null);
    setError("");

    if (inputRef.current) {
      inputRef.current.value = "";
    }

    onFileSelected?.(null);
  }

  return (
    <div className="upload-section">

      <div
        className={
          dragActive
            ? "drop-zone drag-active"
            : "drop-zone"
        }
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >

        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/bmp"
          onChange={handleInputChange}
          hidden
        />

        <div className="upload-icon">
          ↑
        </div>

        <h3>
          Upload an image
        </h3>

        <p>
          Drag and drop your image here
        </p>

        <span className="upload-or">
          or
        </span>

        <button
          type="button"
          className="secondary-button"
          onClick={openFilePicker}
          disabled={loading}
        >
          Choose Image
        </button>

        <div className="upload-hint">
          JPG, PNG, WEBP or BMP · Maximum 10 MB
        </div>

      </div>

      {error && (
        <div className="error-message">
          <span>!</span>
          {error}
        </div>
      )}

      {file && !error && (
        <div className="selected-file">

          <div>
            <strong>
              {file.name}
            </strong>

            <span>
              {(file.size / 1024 / 1024).toFixed(2)} MB
            </span>
          </div>

          <button
            className="remove-button"
            onClick={clearFile}
            disabled={loading}
          >
            Remove
          </button>

        </div>
      )}

      <button
        className="primary-button analyze-button"
        disabled={!file || loading}
        onClick={() => onAnalyze(file)}
      >
        {loading ? (
          <>
            <span className="button-spinner" />
            Analyzing image...
          </>
        ) : (
          <>
            Analyze Image
            <span>→</span>
          </>
        )}
      </button>

    </div>
  );
}

export default ImageUploader;