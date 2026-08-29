import {
    useEffect,
    useCallback,
    useState,
} from "react";

import {
    useNavigate,
} from "react-router-dom";

import Header from "../components/Header";

import {
    getHistory,
    deleteAnalysis,
    API_URL,
} from "../services/api";

import {
    formatDate,
} from "../utils/helper";


function History() {

    const [
        history,
        setHistory,
    ] = useState([]);

    const [
        loading,
        setLoading,
    ] = useState(true);

    const [
        deletingId,
        setDeletingId,
    ] = useState(null);

    const [
        confirmId,
        setConfirmId,
    ] = useState(null);

    const [
        error,
        setError,
    ] = useState(null);


    const navigate =
        useNavigate();


    const loadHistory = useCallback(async () => {

        setLoading(true);
        setError(null);

        try {

            const data =
                await getHistory(20);

            const items =
                Array.isArray(data)
                    ? data
                    : data.items ||
                      data.history ||
                      data.predictions ||
                      [];

            setHistory(items);

        } catch (err) {

            setError(
                err.message ||
                "Unable to load history."
            );

        } finally {

            setLoading(false);

        }
    }, []);


    useEffect(() => {

        void Promise.resolve().then(loadHistory);

    }, [loadHistory]);


    function openAnalysis(
        predictionId
    ) {

        if (!predictionId) {

            setError(
                "This analysis does not have a valid ID."
            );

            return;
        }


        navigate(
            `/results/${predictionId}`
        );
    }


    async function handleDelete(
        predictionId
    ) {

        if (!predictionId) {
            return;
        }


        try {

            setDeletingId(
                predictionId
            );

            setError(null);


            await deleteAnalysis(
                predictionId
            );


            setHistory(
                (prev) =>
                    prev.filter(
                        (item) =>
                            (
                                item.prediction_id ||
                                item._id
                            ) !==
                            predictionId
                    )
            );


            setConfirmId(null);

        } catch (err) {

            setError(
                err.message ||
                "Unable to delete analysis."
            );

        } finally {

            setDeletingId(null);

        }
    }


    function getImageUrl(
        imageUrl
    ) {

        if (!imageUrl) {
            return null;
        }


        if (
            imageUrl.startsWith(
                "http://"
            ) ||
            imageUrl.startsWith(
                "https://"
            )
        ) {

            return imageUrl;
        }


        return `${API_URL}${imageUrl}`;
    }


    return (

        <div className="app">

            <Header />


            <main className="page">

                <div className="results-header">

                    <div>

                        <span className="eyebrow">
                            ANALYSIS HISTORY
                        </span>

                        <h1>
                            Past Analyses
                        </h1>

                        <p>
                            Review or remove previously
                            analyzed images.
                        </p>

                    </div>


                    <button
                        type="button"
                        className="primary-button"
                        onClick={() =>
                            navigate("/")
                        }
                    >
                        New Analysis
                    </button>

                </div>


                {loading && (

                    <div className="loading-state">

                        <div className="spinner" />

                        <p>
                            Loading history...
                        </p>

                    </div>

                )}


                {error && (

                    <div className="error-box">

                        {error}

                    </div>

                )}


                {!loading &&
                    !error &&
                    history.length === 0 && (

                        <div className="empty-state">

                            <div className="empty-icon">
                                ◇
                            </div>

                            <h3>
                                No analyses yet
                            </h3>

                            <p>
                                Your completed image
                                analyses will appear here.
                            </p>

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

                    )}


                {!loading &&
                    history.length > 0 && (

                        <div className="history-list">

                            {history.map(
                                (
                                    item,
                                    index
                                ) => {

                                    const qmos =
                                        Number(
                                            item.qmos ?? 0
                                        );


                                    const score =
                                        Number(
                                            item.quality_score ??
                                            qmos * 20
                                        );


                                    const id =
                                        item.prediction_id ||
                                        item._id;


                                    const imageUrl =
                                        getImageUrl(
                                            item.image_url
                                        );


                                    const isDeleting =
                                        deletingId === id;


                                    const isConfirming =
                                        confirmId === id;


                                    return (

                                        <div
                                            className="history-card"
                                            key={
                                                id ||
                                                index
                                            }
                                        >

                                            <div className="history-image-wrapper">

                                                {imageUrl ? (

                                                    <img
                                                        src={imageUrl}
                                                        alt={
                                                            item.filename ||
                                                            "Analyzed image"
                                                        }
                                                        className="history-image"
                                                        onError={(event) => {
                                                            event.currentTarget.style.display =
                                                                "none";

                                                            event.currentTarget
                                                                .nextElementSibling
                                                                .style.display =
                                                                "flex";
                                                        }}
                                                    />

                                                ) : null}


                                                <div
                                                    className="history-image-placeholder"
                                                    style={{
                                                        display:
                                                            imageUrl
                                                                ? "none"
                                                                : "flex",
                                                    }}
                                                >

                                                    <span>
                                                        🖼️
                                                    </span>

                                                </div>

                                            </div>


                                            <div className="history-info">

                                                <h3 className="history-filename">

                                                    {item.filename ||
                                                        "Unknown image"}

                                                </h3>


                                                <span className="history-date">

                                                    {formatDate(
                                                        item.created_at ||
                                                        item.createdAt ||
                                                        item.timestamp
                                                    )}

                                                </span>


                                                <span
                                                    className={`history-quality ${(
                                                        item.quality_label ||
                                                        ""
                                                    )
                                                        .toLowerCase()
                                                        .replace(
                                                            /\s+/g,
                                                            "-"
                                                        )}`}
                                                >

                                                    {item.quality_label ||
                                                        "Unknown"}

                                                </span>

                                            </div>


                                            <div className="history-metric">

                                                <span>
                                                    qMOS
                                                </span>

                                                <strong>
                                                    {qmos.toFixed(
                                                        2
                                                    )}
                                                </strong>

                                                <small>
                                                    / 5
                                                </small>

                                            </div>


                                            <div className="history-metric">

                                                <span>
                                                    Quality
                                                </span>

                                                <strong>
                                                    {score.toFixed(
                                                        1
                                                    )}
                                                </strong>

                                                <small>
                                                    / 100
                                                </small>

                                            </div>


                                            <div className="history-actions">

                                                <button
                                                    type="button"
                                                    className="history-view-button"
                                                    onClick={() =>
                                                        openAnalysis(
                                                            id
                                                        )
                                                    }
                                                    disabled={
                                                        isDeleting
                                                    }
                                                >

                                                    View →

                                                </button>


                                                {isConfirming ? (

                                                    <div className="history-delete-confirm">

                                                        <button
                                                            type="button"
                                                            className="history-delete-confirm-btn"
                                                            onClick={() =>
                                                                handleDelete(
                                                                    id
                                                                )
                                                            }
                                                            disabled={
                                                                isDeleting
                                                            }
                                                        >

                                                            {isDeleting
                                                                ? "Deleting..."
                                                                : "Confirm"}

                                                        </button>


                                                        <button
                                                            type="button"
                                                            className="history-delete-cancel-btn"
                                                            onClick={() =>
                                                                setConfirmId(
                                                                    null
                                                                )
                                                            }
                                                            disabled={
                                                                isDeleting
                                                            }
                                                        >

                                                            Cancel

                                                        </button>

                                                    </div>

                                                ) : (

                                                    <button
                                                        type="button"
                                                        className="history-delete-button"
                                                        onClick={() =>
                                                            setConfirmId(
                                                                id
                                                            )
                                                        }
                                                        disabled={
                                                            isDeleting
                                                        }
                                                        aria-label="Delete analysis"
                                                    >

                                                        Delete

                                                    </button>

                                                )}

                                            </div>

                                        </div>

                                    );

                                }
                            )}

                        </div>

                    )}

            </main>

        </div>
    );
}


export default History;
