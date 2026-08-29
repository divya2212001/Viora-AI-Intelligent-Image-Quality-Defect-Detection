function HeatmapViewer({
    heatmapUrl,
}) {

    if (!heatmapUrl) {

        return (
            <div className="heatmap-empty">

                <div className="heatmap-icon">
                    ◎
                </div>

                <h3>
                    Grad-CAM unavailable
                </h3>

                <p>
                    No explanation image was
                    generated for this analysis.
                </p>

            </div>
        );
    }


    return (
        <div className="heatmap-viewer">

            <img
                src={heatmapUrl}
                alt="Grad-CAM explanation"
                className="heatmap-image"
            />

            <p className="heatmap-description">
                Highlighted regions show the
                image areas that contributed
                to the predicted quality score.
            </p>

        </div>
    );
}


export default HeatmapViewer;