import { useState } from "react";

import Header from "./components/Header";
import Home from "./pages/Home";
import Results from "./pages/Results";
import History from "./pages/History";

import { useAnalysis } from "./hooks/useAnalysis";

function App() {
  const [currentPage, setCurrentPage] =
    useState("home");

  const [imageUrl, setImageUrl] =
    useState(null);

  const [selectedHistoryResult, setSelectedHistoryResult] =
    useState(null);

  const {
    result,
    loading,
    error,
    runAnalysis,
    clearResult,
  } = useAnalysis();

  async function handleAnalyze(file) {
    if (!file) return;

    if (imageUrl) {
      URL.revokeObjectURL(imageUrl);
    }

    const localImageUrl =
      URL.createObjectURL(file);

    setImageUrl(localImageUrl);
    setSelectedHistoryResult(null);

    try {
      await runAnalysis(file);

      setCurrentPage("results");
    } catch {
      // Error is handled by useAnalysis.
    }
  }

  function handleNewAnalysis() {
    clearResult();
    setSelectedHistoryResult(null);

    if (imageUrl) {
      URL.revokeObjectURL(imageUrl);
    }

    setImageUrl(null);
    setCurrentPage("home");
  }

  function handleViewHistoryAnalysis(analysis) {
    setSelectedHistoryResult(analysis);
    setCurrentPage("results");
  }

  function navigate(page) {
    setCurrentPage(page);
  }

  const displayedResult =
    selectedHistoryResult || result;

  const displayedImage =
    selectedHistoryResult?.image_url ||
    selectedHistoryResult?.imageUrl ||
    imageUrl;

  return (
    <div className="app">

      <Header
        currentPage={currentPage}
        onNavigate={navigate}
      />

      {currentPage === "home" && (
        <Home
          onAnalyze={handleAnalyze}
          loading={loading}
          error={error}
        />
      )}

      {currentPage === "results" && (
        <Results
          result={displayedResult}
          imageUrl={displayedImage}
          onNewAnalysis={handleNewAnalysis}
        />
      )}

      {currentPage === "history" && (
        <History
          onViewAnalysis={handleViewHistoryAnalysis}
        />
      )}

      <footer className="app-footer">

        <p>
          Viora AI · Image Quality & Defect Detection
        </p>

        <span>
          AI-powered · Computer Vision · Local Inference
        </span>

      </footer>

    </div>
  );
}

export default App;