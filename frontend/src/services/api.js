const API_URL =
    import.meta.env.VITE_API_BASE_URL ||
    "http://127.0.0.1:8000";

const SESSION_ID_KEY =
    "viora-analysis-session-id";


function getSessionId() {

    let sessionId =
        sessionStorage.getItem(
            SESSION_ID_KEY
        );

    if (!sessionId) {

        sessionId = crypto.randomUUID();

        sessionStorage.setItem(
            SESSION_ID_KEY,
            sessionId
        );
    }

    return sessionId;
}


function withSessionId(path) {

    const separator =
        path.includes("?")
            ? "&"
            : "?";

    return (
        `${path}${separator}session_id=` +
        encodeURIComponent(getSessionId())
    );
}


export async function predictImage(file) {

    const formData = new FormData();

    formData.append(
        "file",
        file
    );


    const response = await fetch(
        `${API_URL}${withSessionId("/api/predict")}`,
        {
            method: "POST",
            body: formData,
        }
    );


    if (!response.ok) {

        const error =
            await response.text();

        throw new Error(
            error || "Prediction failed"
        );
    }


    return response.json();
}


export async function getHistory(
    limit = 20
) {

    const response = await fetch(
        `${API_URL}${withSessionId(
            `/api/history?limit=${limit}`
        )}`
    );


    if (!response.ok) {

        throw new Error(
            "Failed to load history"
        );
    }


    return response.json();
}


export async function getAnalysis(
    analysisId
) {

    const response = await fetch(
        `${API_URL}${withSessionId(
            `/api/analyses/${analysisId}`
        )}`
    );


    if (!response.ok) {

        throw new Error(
            "Failed to load analysis"
        );
    }


    return response.json();
}


export async function deleteAnalysis(
    analysisId
) {

    const response = await fetch(
        `${API_URL}${withSessionId(
            `/api/analyses/${analysisId}`
        )}`,
        {
            method: "DELETE",
        }
    );


    if (!response.ok) {

        throw new Error(
            "Failed to delete analysis"
        );
    }


    return response.json();
}



export async function getModelInfo() {

    const response = await fetch(
        `${API_URL}/api/model-info`
    );


    if (!response.ok) {

        throw new Error(
            "Failed to load model information"
        );
    }


    return response.json();
}

export {
    API_URL,
};
