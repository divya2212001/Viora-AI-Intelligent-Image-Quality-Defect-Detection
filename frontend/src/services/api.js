const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function handleResponse(response) {
  let data = null;

  try {
    data = await response.json();
  } catch {
    data = null;
  }

  if (!response.ok) {
    const message =
      data?.detail ||
      data?.message ||
      `Request failed with status ${response.status}`;

    throw new Error(message);
  }

  return data;
}

export async function analyzeImage(file) {
  const formData = new FormData();

  formData.append("file", file);

  const response = await fetch(
    `${API_BASE_URL}/api/analyze`,
    {
      method: "POST",
      body: formData,
    }
  );

  return handleResponse(response);
}

export async function getAnalyses() {
  const response = await fetch(
    `${API_BASE_URL}/api/analyses`
  );

  return handleResponse(response);
}

export async function getAnalysis(id) {
  const response = await fetch(
    `${API_BASE_URL}/api/analyses/${id}`
  );

  return handleResponse(response);
}

export async function deleteAnalysis(id) {
  const response = await fetch(
    `${API_BASE_URL}/api/analyses/${id}`,
    {
      method: "DELETE",
    }
  );

  return handleResponse(response);
}

export async function analyzeBatch(files) {
  const formData = new FormData();

  files.forEach((file) => {
    formData.append("files", file);
  });

  const response = await fetch(
    `${API_BASE_URL}/api/analyze/batch`,
    {
      method: "POST",
      body: formData,
    }
  );

  return handleResponse(response);
}

export async function getHealth() {
  const response = await fetch(
    `${API_BASE_URL}/health`
  );

  return handleResponse(response);
}