import { useEffect, useState } from "react";
import {
  deleteAnalysis,
  getAnalyses,
  getAnalysis,
} from "../services/api";
import {
  formatDate,
  getQualityClass,
} from "../utils/helpers";
import EmptyState from "../components/EmptyState";

function History({ onViewAnalysis }) {
  const [analyses, setAnalyses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  async function loadHistory() {
    setLoading(true);
    setError("");

    try {
      const data = await getAnalyses();

      if (Array.isArray(data)) {
        setAnalyses(data);
      } else {
        setAnalyses(data.analyses || []);
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to load analysis history."
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadHistory();
  }, []);

  async function handleDelete(id) {
    const confirmed = window.confirm(
      "Delete this analysis?"
    );

    if (!confirmed) {
      return;
    }

    try {
      await deleteAnalysis(id);

      setAnalyses((previous) =>
        previous.filter(
          (analysis) =>
            (analysis.id || analysis._id) !== id
        )
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to delete analysis."
      );
    }
  }

  async function handleView(id) {
    try {
      const analysis = await getAnalysis(id);

      onViewAnalysis(analysis);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to load analysis."
      );
    }
  }

  if (loading) {
    return (
      <main className="page-container">
        <div className="loading-card">
          <div className="large-spinner" />

          <h2>
            Loading history
          </h2>

          <p>
            Retrieving previous analyses...
          </p>
        </div>
      </main>
    );
  }

  return (
    <main className="page-container">

      <div className="results-heading">

        <div>
          <div className="section-label">
            ANALYSIS HISTORY
          </div>

          <h1>
            Previous analyses
          </h1>

          <p>
            Review images that have been analyzed
            previously.
          </p>
        </div>

        <button
          className="secondary-button"
          onClick={loadHistory}
        >
          ↻ Refresh
        </button>

      </div>

      {error && (
        <div className="error-message large">
          <span>!</span>
          {error}
        </div>
      )}

      {analyses.length === 0 ? (
        <EmptyState
          title="No analyses yet"
          description="Your completed image analyses will appear here."
        />
      ) : (
        <div className="history-card">

          <div className="history-header">
            <span>IMAGE</span>
            <span>SCORE</span>
            <span>STATUS</span>
            <span>DATE</span>
            <span>ACTIONS</span>
          </div>

          {analyses.map((analysis) => {
            const id =
              analysis.id || analysis._id;

            return (
              <div
                className="history-row"
                key={id}
              >

                <div className="history-image-info">

                  <div className="history-thumbnail">
                    {analysis.image_url ? (
                      <img
                        src={analysis.image_url}
                        alt=""
                      />
                    ) : (
                      <span>IMG</span>
                    )}
                  </div>

                  <div>
                    <strong>
                      {analysis.filename ||
                        "Unnamed image"}
                    </strong>

                    <small>
                      ID: {String(id).slice(0, 10)}
                    </small>
                  </div>

                </div>

                <div className="history-score">
                  {analysis.quality_score ?? "--"}
                  <small>/100</small>
                </div>

                <div>
                  <span
                    className={`quality-badge small ${getQualityClass(
                      analysis.quality_label
                    )}`}
                  >
                    {analysis.quality_label ||
                      "UNKNOWN"}
                  </span>
                </div>

                <div className="history-date">
                  {formatDate(
                    analysis.created_at
                  )}
                </div>

                <div className="history-actions">

                  <button
                    className="text-button"
                    onClick={() =>
                      handleView(id)
                    }
                  >
                    View
                  </button>

                  <button
                    className="delete-button"
                    onClick={() =>
                      handleDelete(id)
                    }
                  >
                    Delete
                  </button>

                </div>

              </div>
            );
          })}

        </div>
      )}

    </main>
  );
}

export default History;