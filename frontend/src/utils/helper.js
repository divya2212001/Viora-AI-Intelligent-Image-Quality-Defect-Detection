export function formatDate(dateString) {
  if (!dateString) return "Unknown";

  const date = new Date(dateString);

  if (Number.isNaN(date.getTime())) {
    return "Unknown";
  }

  return date.toLocaleString();
}

export function formatConfidence(value) {
  if (value === null || value === undefined) {
    return "N/A";
  }

  const number = Number(value);

  if (Number.isNaN(number)) {
    return "N/A";
  }

  return `${(number * 100).toFixed(1)}%`;
}

export function getSeverityClass(severity) {
  if (!severity) return "severity-unknown";

  switch (severity.toLowerCase()) {
    case "low":
      return "severity-low";

    case "medium":
      return "severity-medium";

    case "high":
      return "severity-high";

    case "critical":
      return "severity-critical";

    default:
      return "severity-unknown";
  }
}

export function getQualityClass(label) {
  if (!label) return "quality-default";

  switch (label.toUpperCase()) {
    case "EXCELLENT":
      return "quality-excellent";

    case "ACCEPTABLE":
      return "quality-acceptable";

    case "DEGRADED":
      return "quality-degraded";

    case "DEFECTIVE":
      return "quality-defective";

    default:
      return "quality-default";
  }
}

export function formatIssueName(type) {
  if (!type) return "Unknown issue";

  return type
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

export function getScoreDescription(score) {
  if (score >= 90) {
    return "Excellent visual quality";
  }

  if (score >= 75) {
    return "Good quality with minor concerns";
  }

  if (score >= 50) {
    return "Image quality needs attention";
  }

  return "Significant quality problems detected";
}