const PYTHON_API_URL = import.meta.env.VITE_PYTHON_API_URL

export async function analyzePos(text){
    const response = await fetch(PYTHON_API_URL + '/pos',{
        method:"POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({text})
    })

    return response.json()
}