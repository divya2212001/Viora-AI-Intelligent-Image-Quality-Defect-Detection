import { useState } from "react";
import { analyzeImage } from "../services/api";

export function useAnalysis() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function runAnalysis(file) {
    setLoading(true);
    setError(null);

    try {
      const data = await analyzeImage(file);

      setResult(data);

      return data;
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Unable to analyze image.";

      setError(message);

      throw err;
    } finally {
      setLoading(false);
    }
  }

  function clearResult() {
    setResult(null);
    setError(null);
  }

  return {
    result,
    loading,
    error,
    runAnalysis,
    clearResult,
  };
}