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

            <div className="heatmap-image-container">

                <img
                    src={heatmapUrl}
                    alt="Grad-CAM explanation"
                    className="heatmap-image"
                    onError={(event) => {

                        console.error(
                            "Failed to load Grad-CAM:",
                            heatmapUrl
                        );

                        event.currentTarget.style.display =
                            "none";

                        const errorElement =
                            event.currentTarget
                                .nextElementSibling;

                        if (errorElement) {

                            errorElement.style.display =
                                "flex";
                        }
                    }}
                />


                <div
                    className="heatmap-load-error"
                    style={{
                        display: "none",
                    }}
                >

                    <div className="heatmap-icon">
                        ⚠
                    </div>

                    <h3>
                        Unable to load Grad-CAM
                    </h3>

                    <p>
                        The explanation image could
                        not be loaded from the server.
                    </p>

                </div>

            </div>


            <p className="heatmap-description">

                Highlighted regions show the
                image areas that contributed
                to the predicted quality score.

            </p>

        </div>
    );
}


export default HeatmapViewer;