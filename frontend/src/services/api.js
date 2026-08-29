const API_URL =
    import.meta.env.API_URL ||
    "http://127.0.0.1:8000";



// SESSION ID


function getSessionId() {

    let sessionId =
        localStorage.getItem(
            "viora_session_id"
        );

    if (!sessionId) {

        sessionId =
            crypto.randomUUID();

        localStorage.setItem(
            "viora_session_id",
            sessionId
        );
    }

    return sessionId;
}



// PREDICT IMAGE


export async function predictImage(file) {

    const formData =
        new FormData();

    formData.append(
        "file",
        file
    );

    const sessionId =
        getSessionId();

    const response =
        await fetch(
            `${API_URL}/api/predict?session_id=${encodeURIComponent(
                sessionId
            )}`,
            {
                method: "POST",
                body: formData,
            }
        );


    if (!response.ok) {

        const error =
            await response.text();

        throw new Error(
            error ||
            "Prediction failed"
        );
    }


    return response.json();
}



// HISTORY


export async function getHistory(
    limit = 20
) {

    const sessionId =
        getSessionId();

    const response =
        await fetch(
            `${API_URL}/api/history?limit=${limit}&session_id=${encodeURIComponent(
                sessionId
            )}`
        );


    if (!response.ok) {

        const error =
            await response.text();

        throw new Error(
            error ||
            "Failed to load history"
        );
    }


    return response.json();
}



// GET SINGLE ANALYSIS


export async function getAnalysis(
    analysisId
) {

    const sessionId =
        getSessionId();

    const response =
        await fetch(
            `${API_URL}/api/analyses/${encodeURIComponent(
                analysisId
            )}?session_id=${encodeURIComponent(
                sessionId
            )}`
        );


    if (!response.ok) {

        const error =
            await response.text();

        throw new Error(
            error ||
            "Failed to load analysis"
        );
    }


    return response.json();
}



// DELETE ANALYSIS


export async function deleteAnalysis(
    analysisId
) {

    const sessionId =
        getSessionId();

    const response =
        await fetch(
            `${API_URL}/api/analyses/${encodeURIComponent(
                analysisId
            )}?session_id=${encodeURIComponent(
                sessionId
            )}`,
            {
                method: "DELETE",
            }
        );


    if (!response.ok) {

        const error =
            await response.text();

        throw new Error(
            error ||
            "Failed to delete analysis"
        );
    }


    return response.json();
}



// MODEL INFO


export async function getModelInfo() {

    const response =
        await fetch(
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
    getSessionId,
};