import QualityScore from "../components/QualityScore";
import IssueList from "../components/IssueList";
import Statistics from "../components/Statistics";
import ModelInfo from "../components/ModelInfo";
import HeatmapViewer from "../components/HeatmapViewer";

function Results({
  result,
  imageUrl,
  onNewAnalysis,
}) {
  if (!result) {
    return (
      <main className="page-container">
        <div className="empty-state">
          <h2>
            No analysis available
          </h2>

          <p>
            Upload an image to start an analysis.
          </p>

          <button
            className="primary-button"
            onClick={onNewAnalysis}
          >
            Analyze an image
          </button>
        </div>
      </main>
    );
  }

  return (
    <main className="page-container results-page">

      <div className="results-heading">

        <div>
          <div className="section-label">
            ANALYSIS RESULT
          </div>

          <h1>
            Image quality report
          </h1>

          <p>
            AI and computer-vision analysis of your
            uploaded image.
          </p>
        </div>

        <button
          className="secondary-button"
          onClick={onNewAnalysis}
        >
          ← Analyze another
        </button>

      </div>

      <div className="results-grid">

        <div className="results-image-column">

          {imageUrl && (
            <div className="result-image-card">

              <div className="section-label">
                ANALYZED IMAGE
              </div>

              <img
                src={imageUrl}
                alt="Analyzed image"
                className="result-image"
              />

            </div>
          )}

          <ModelInfo model={result.model} />

        </div>

        <div className="results-content">

          <QualityScore
            score={result.quality_score}
            label={result.quality_label}
          />

          <IssueList
            issues={result.issues || []}
          />

          <Statistics
            statistics={result.statistics || {}}
          />

          <HeatmapViewer
            heatmap={
              result.explainability?.heatmap ||
              result.heatmap
            }
          />

        </div>

      </div>

    </main>
  );
}

export default Results;