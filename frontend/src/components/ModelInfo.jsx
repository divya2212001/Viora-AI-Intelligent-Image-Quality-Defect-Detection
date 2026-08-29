function ModelInfo({ model }) {
  if (!model) {
    return null;
  }

  return (
    <div className="model-info">

      <div>
        <span>MODEL</span>
        <strong>
          {model.name || "Unknown"}
        </strong>
      </div>

      <div>
        <span>VERSION</span>
        <strong>
          {model.version || "Unknown"}
        </strong>
      </div>

    </div>
  );
}

export default ModelInfo;