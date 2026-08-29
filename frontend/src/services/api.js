const API_URL = "http://127.0.0.1:8000";


/*
 * ==========================================
 * PREDICT IMAGE
 * ==========================================
 */

export async function predictImage(file) {

    const formData = new FormData();

    formData.append(
        "file",
        file
    );


    const response = await fetch(
        `${API_URL}/api/predict`,
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



/*
 * ==========================================
 * GET HISTORY
 * ==========================================
 */

export async function getHistory(
    limit = 20
) {

    const response = await fetch(
        `${API_URL}/api/history?limit=${limit}`
    );


    if (!response.ok) {

        throw new Error(
            "Failed to load history"
        );
    }


    return response.json();
}



/*
 * ==========================================
 * GET SINGLE ANALYSIS
 * ==========================================
 */

export async function getAnalysis(
    analysisId
) {

    const response = await fetch(
        `${API_URL}/api/analyses/${analysisId}`
    );


    if (!response.ok) {

        throw new Error(
            "Failed to load analysis"
        );
    }


    return response.json();
}



/*
 * ==========================================
 * GET MODEL INFORMATION
 * ==========================================
 */

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



/*
 * ==========================================
 * API URL
 * ==========================================
 */

export {
    API_URL,
};