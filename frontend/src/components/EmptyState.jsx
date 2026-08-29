function EmptyState({
  title = "No analyses yet",
  description = "Upload an image to start your first analysis.",
}) {
  return (
    <div className="empty-state">

      <div className="empty-icon">
        ◇
      </div>

      <h2>
        {title}
      </h2>

      <p>
        {description}
      </p>

    </div>
  );
}

export default EmptyState;