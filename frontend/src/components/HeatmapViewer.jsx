function HeatmapViewer({ heatmap }) {
  if (!heatmap) {
    return null;
  }

  let source = heatmap;

  if (
    typeof heatmap === "string" &&
    heatmap.startsWith("data:")
  ) {
    source = heatmap;
  }

  return (
    <div className="card">

      <div className="section-label">
        EXPLAINABILITY
      </div>

      <h2>
        Problem localization
      </h2>

      <p className="card-description">
        Regions highlighted by the model that
        contributed to the quality assessment.
      </p>

      <div className="heatmap-container">
        <img
          src={source}
          alt="Quality analysis heatmap"
          className="heatmap-image"
        />
      </div>

    </div>
  );
}

export default HeatmapViewer;