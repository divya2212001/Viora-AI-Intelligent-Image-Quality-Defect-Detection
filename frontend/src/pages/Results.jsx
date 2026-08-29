import {
    useLocation,
    useNavigate,
} from "react-router-dom";

import Header from "../components/Header";
import QualityScore from "../components/QualityScore";
import IssueList from "../components/IssueList";
import Statistics from "../components/Statistics";
import HeatmapViewer from "../components/HeatmapViewer";
import EmptyState from "../components/EmptyState";

import { API_URL } from "../services/api";


function Results() {

    const location =
        useLocation();

    const navigate =
        useNavigate();


    const result =
        location.state?.result;


    /*
     * ------------------------------------------------
     * NO RESULT
     * ------------------------------------------------
     */

    if (!result) {

        return (

            <div className="app">

                <Header />

                <main className="page">

                    <EmptyState
                        title="No analysis found"
                        description="Upload an image first to view analysis results."
                    />

                    <button
                        className="primary-button"
                        onClick={() =>
                            navigate("/")
                        }
                    >
                        Analyze an Image
                    </button>

                </main>

            </div>
        );
    }


    /*
     * ------------------------------------------------
     * IMAGE URL
     * ------------------------------------------------
     */

    const imageUrl =
        result.image_url
            ? `${API_URL}${result.image_url}`
            : null;


    /*
     * ------------------------------------------------
     * GRAD-CAM URL
     * ------------------------------------------------
     */

    const gradcamUrl =
        result.gradcam_url
            ? `${API_URL}${result.gradcam_url}`
            : null;


    /*
     * ------------------------------------------------
     * PAGE
     * ------------------------------------------------
     */

    return (

        <div className="app">

            <Header />


            <main className="page results-page">


                {/* =====================================
                    HEADER
                ===================================== */}

                <div className="results-header">

                    <div>

                        <span className="eyebrow">
                            ANALYSIS COMPLETE
                        </span>

                        <h1>
                            Analysis Results
                        </h1>

                        <p>
                            {result.filename}
                        </p>

                    </div>


                    <button
                        className="secondary-button"
                        onClick={() =>
                            navigate("/")
                        }
                    >
                        Analyze Another
                    </button>

                </div>



                {/* =====================================
                    ORIGINAL IMAGE
                ===================================== */}

                {imageUrl && (

                    <section className="image-card">

                        <div className="image-card-header">

                            <div>

                                <span className="eyebrow">
                                    ANALYZED IMAGE
                                </span>

                                <h2>
                                    Uploaded Image
                                </h2>

                            </div>

                        </div>


                        <div className="original-image-container">

                            <img
                                src={imageUrl}
                                alt={
                                    result.filename ||
                                    "Analyzed image"
                                }
                                className="original-image"
                            />

                        </div>

                    </section>

                )}



                {/* =====================================
                    QUALITY SCORE
                ===================================== */}

                <QualityScore
                    qmos={result.qmos}
                    qualityScore={
                        result.quality_score
                    }
                    qualityLabel={
                        result.quality_label
                    }
                />



                {/* =====================================
                    ISSUES + RECOMMENDATION
                ===================================== */}

                <div className="results-grid">


                    <IssueList
                        defects={
                            result.defects || {}
                        }
                    />


                    <section className="recommendation-card">

                        <span className="eyebrow">
                            RECOMMENDATION
                        </span>

                        <h2>
                            What the model says
                        </h2>

                        <p>
                            {result.recommendation ||
                                "No recommendation available."}
                        </p>

                    </section>


                </div>



                {/* =====================================
                    IMAGE STATISTICS
                ===================================== */}

                <Statistics
                    statistics={
                        result.statistics || {}
                    }
                />



                {/* =====================================
                    GRAD-CAM
                ===================================== */}

                <section className="gradcam-section">

                    <div className="section-header">

                        <span className="eyebrow">
                            EXPLAINABILITY · BONUS
                        </span>

                        <h2>
                            Grad-CAM Explainability
                        </h2>

                        <p>
                            Visual explanation of the
                            image regions that influenced
                            the model's quality prediction.
                        </p>

                    </div>


                    <HeatmapViewer
                        heatmapUrl={
                            gradcamUrl
                        }
                    />

                </section>


            </main>

        </div>
    );
}


export default Results;