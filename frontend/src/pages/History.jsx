import {
    useEffect,
    useState,
} from "react";

import {
    useNavigate,
} from "react-router-dom";

import Header from "../components/Header";

import {
    getHistory,
    getAnalysis,
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
        selectedId,
        setSelectedId,
    ] = useState(null);

    const [
        error,
        setError,
    ] = useState(null);


    const navigate =
        useNavigate();


    useEffect(() => {

        loadHistory();

    }, []);


    async function loadHistory() {

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
    }


    async function openAnalysis(
        predictionId
    ) {

        if (!predictionId) {
            return;
        }

        try {

            setSelectedId(
                predictionId
            );

            const result =
                await getAnalysis(
                    predictionId
                );

            navigate(
                "/results",
                {
                    state: {
                        result,
                        fromHistory: true,
                    },
                }
            );

        } catch (err) {

            setError(
                err.message ||
                "Unable to open analysis."
            );

        } finally {

            setSelectedId(null);
        }
    }


    return (

        <div className="app">

            <Header />


            <main className="page">

                <div className="results-header">

                    <div>

                        <span className="eyebrow">
                            MONGODB HISTORY
                        </span>

                        <h1>
                            Analysis History
                        </h1>

                        <p>
                            Previously analyzed images.
                            Click any analysis to view
                            the complete summary.
                        </p>

                    </div>


                    <button
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

                            <h3>
                                No analyses yet
                            </h3>

                            <p>
                                Your completed image
                                analyses will appear here.
                            </p>

                        </div>

                    )}


                <div className="history-list">

                    {history.map(
                        (item, index) => {

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


                            return (

                                <button
                                    className="history-card"
                                    key={
                                        id ||
                                        index
                                    }
                                    onClick={() =>
                                        openAnalysis(
                                            id
                                        )
                                    }
                                    disabled={
                                        selectedId === id
                                    }
                                >

                                    <div>

                                        <span className="history-filename">
                                            {item.filename ||
                                                "Unknown image"}
                                        </span>

                                        <span className="history-date">
                                            {formatDate(
                                                item.created_at ||
                                                item.createdAt ||
                                                item.timestamp
                                            )}
                                        </span>

                                    </div>


                                    <div className="history-score">

                                        <strong>
                                            {qmos.toFixed(
                                                2
                                            )}
                                        </strong>

                                        <span>
                                            / 5
                                        </span>

                                    </div>


                                    <div className="history-score">

                                        <strong>
                                            {score.toFixed(
                                                1
                                            )}
                                        </strong>

                                        <span>
                                            / 100
                                        </span>

                                    </div>


                                    <span className="history-open">

                                        {selectedId === id
                                            ? "Opening..."
                                            : "View →"}

                                    </span>

                                </button>

                            );

                        }
                    )}

                </div>

            </main>

        </div>
    );
}


export default History;