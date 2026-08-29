import {
  formatConfidence,
  formatIssueName,
  getSeverityClass,
} from "../utils/helpers";

function IssueList({ issues = [] }) {
  return (
    <div className="card">

      <div className="card-header">

        <div>
          <div className="section-label">
            DETECTED ISSUES
          </div>

          <h2>
            Quality concerns
          </h2>
        </div>

        <div className="issue-count">
          {issues.length}
        </div>

      </div>

      {issues.length === 0 ? (
        <div className="no-issues">
          <div className="success-icon">
            ✓
          </div>

          <div>
            <strong>
              No significant issues detected
            </strong>

            <p>
              The model did not identify any major
              quality concerns.
            </p>
          </div>
        </div>
      ) : (
        <div className="issue-list">

          {issues.map((issue, index) => (
            <div
              className="issue-row"
              key={`${issue.type}-${index}`}
            >

              <div className="issue-main">

                <div className="issue-icon">
                  !
                </div>

                <div>
                  <strong>
                    {formatIssueName(issue.type)}
                  </strong>

                  <span>
                    Confidence{" "}
                    {formatConfidence(issue.confidence)}
                  </span>
                </div>

              </div>

              <div
                className={`severity-badge ${getSeverityClass(
                  issue.severity
                )}`}
              >
                {issue.severity || "unknown"}
              </div>

            </div>
          ))}

        </div>
      )}

    </div>
  );
}

export default IssueList;