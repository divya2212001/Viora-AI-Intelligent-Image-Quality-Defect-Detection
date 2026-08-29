import { useState } from "react";
import ImageUploader from "../components/ImageUploader";
import ImagePreview from "../components/ImagePreview";

function Home({
  onAnalyze,
  loading,
  error,
}) {
  const [selectedFile, setSelectedFile] = useState(null);

  const previewUrl = selectedFile
    ? URL.createObjectURL(selectedFile)
    : null;

  return (
    <main className="page-container">

      <section className="hero">

        <div className="hero-badge">
          AI-POWERED COMPUTER VISION
        </div>

        <h1>
          Understand your image
          <br />
          <span>before you use it.</span>
        </h1>

        <p>
          Analyze sharpness, exposure, noise, contrast,
          degradation and potential visual defects using
          computer vision and machine learning.
        </p>

      </section>

      <section className="workspace">

        <div className="upload-panel">

          <div className="panel-header">
            <div>
              <div className="section-label">
                IMAGE ANALYSIS
              </div>

              <h2>
                Upload an image
              </h2>
            </div>
          </div>

          <ImageUploader
            onFileSelected={setSelectedFile}
            onAnalyze={onAnalyze}
            loading={loading}
          />

          {error && (
            <div className="error-message large">
              <span>!</span>
              {error}
            </div>
          )}

        </div>

        <div className="preview-panel">

          {previewUrl ? (
            <ImagePreview
              src={previewUrl}
              alt="Selected image"
            />
          ) : (
            <div className="preview-placeholder">

              <div className="placeholder-icon">
                ◌
              </div>

              <h3>
                Image preview
              </h3>

              <p>
                Your selected image will appear here.
              </p>

            </div>
          )}

        </div>

      </section>

      <section className="capabilities">

        <div className="capability">
          <span>01</span>
          <strong>Sharpness</strong>
          <p>Blur and insufficient focus detection.</p>
        </div>

        <div className="capability">
          <span>02</span>
          <strong>Exposure</strong>
          <p>Underexposure and overexposure analysis.</p>
        </div>

        <div className="capability">
          <span>03</span>
          <strong>Noise</strong>
          <p>Detection of unwanted image noise.</p>
        </div>

        <div className="capability">
          <span>04</span>
          <strong>Defects</strong>
          <p>Potential degradation and visual defects.</p>
        </div>

      </section>

    </main>
  );
}

export default Home;