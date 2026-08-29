function ImagePreview({ src, alt = "Uploaded image" }) {
  if (!src) {
    return null;
  }

  return (
    <div className="image-preview-card">

      <div className="section-label">
        INPUT IMAGE
      </div>

      <div className="image-preview-container">
        <img
          src={src}
          alt={alt}
          className="image-preview"
        />
      </div>

    </div>
  );
}

export default ImagePreview;