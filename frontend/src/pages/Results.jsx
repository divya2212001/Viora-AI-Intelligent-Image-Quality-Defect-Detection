import {
    useEffect,
    useState,
} from "react";

import {
    useLocation,
    useNavigate,
    useParams,
} from "react-router-dom";

import Header from "../components/Header";
import QualityScore from "../components/QualityScore";
import IssueList from "../components/IssueList";
import Statistics from "../components/Statistics";
import HeatmapViewer from "../components/HeatmapViewer";
import EmptyState from "../components/EmptyState";

import {
    API_URL,
    getAnalysis,
} from "../services/api";


function Results() {

    const location =
        useLocation();

    const navigate =
        useNavigate();

    const { id } =
        useParams();


    const [
        result,
        setResult,
    ] = useState(
        location.state?.result ?? null
    );

    const [
        loading,
        setLoading,
    ] = useState(
        !location.state?.result &&
        Boolean(id)
    );

    const [
        error,
        setError,
    ] = useState(null);


    useEffect(() => {

        if (location.state?.result) {

            setResult(
                location.state.result
            );

            return;
        }


        if (!id) {
            return;
        }


        let cancelled = false;


        async function loadResult() {

            setLoading(true);
            setError(null);


            try {

                const data =
                    await getAnalysis(id);

                if (!cancelled) {
                    setResult(data);
                }

            } catch (err) {

                if (!cancelled) {

                    setError(
                        err.message ||
                        "Unable to load analysis."
                    );

                }

            } finally {

                if (!cancelled) {
                    setLoading(false);
                }

            }
        }


        loadResult();


        return () => {
            cancelled = true;
        };

    }, [
        id,
        location.state?.result,
    ]);


    function buildUrl(
        path
    ) {

        if (!path) {
            return null;
        }


        if (
            path.startsWith(
                "http://"
            ) ||
            path.startsWith(
                "https://"
            )
        ) {

            return path;
        }


        return `${API_URL}${path}`;
    }


    if (loading) {

        return (

            <div className="app">

                <Header />

                <main className="page">

                    <div className="loading-state">

                        <div className="spinner" />

                        <p>
                            Loading analysis...
                        </p>

                    </div>

                </main>

            </div>
        );
    }


    if (error || !result) {

        return (

            <div className="app">

                <Header />

                <main className="page">

                    <EmptyState
                        title={
                            error
                                ? "Analysis not found"
                                : "No analysis found"
                        }
                        description={
                            error ||
                            "Upload an image first to view analysis results."
                        }
                    />

                    <div className="empty-actions">

                        <button
                            type="button"
                            className="secondary-button"
                            onClick={() =>
                                navigate("/history")
                            }
                        >
                            View History
                        </button>

                        <button
                            type="button"
                            className="primary-button"
                            onClick={() =>
                                navigate("/")
                            }
                        >
                            Analyze an Image
                        </button>

                    </div>

                </main>

            </div>
        );
    }


    const imageUrl =
        buildUrl(
            result.image_url
        );


    const gradcamUrl =
        buildUrl(
            result.gradcam_url
        );


    return (

        <div className="app">

            <Header />


            <main className="page results-page">


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


                    <div className="results-actions">

                        <button
                            type="button"
                            className="secondary-button"
                            onClick={() =>
                                navigate("/history")
                            }
                        >
                            ← History
                        </button>


                        <button
                            type="button"
                            className="primary-button"
                            onClick={() =>
                                navigate("/")
                            }
                        >
                            Analyze Another
                        </button>

                    </div>

                </div>


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

                        {imageUrl ? (

                            <img
                                src={imageUrl}
                                alt={
                                    result.filename ||
                                    "Analyzed image"
                                }
                                className="original-image"
                            />

                        ) : (

                            <div className="image-not-available">

                                <span>
                                    🖼️
                                </span>

                                <p>
                                    Original image is
                                    unavailable.
                                </p>

                            </div>

                        )}

                    </div>

                </section>


                <QualityScore
                    qmos={
                        result.qmos
                    }

                    qualityScore={
                        result.quality_score
                    }

                    qualityLabel={
                        result.quality_label
                    }
                />


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


                <Statistics
                    statistics={
                        result.statistics || {}
                    }
                />


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
