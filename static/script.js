function submitReview() {
    const review = document.getElementById('review').value.trim();
    const resultDiv = document.getElementById('result');

    if (review === "") {
        resultDiv.innerHTML = `<p style="color:red;">Please enter a review first.</p>`;
        return;
    }

    fetch('/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review })
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            resultDiv.innerHTML = `<p style="color:red;">${data.error}</p>`;
            return;
        }
        resultDiv.innerHTML = `
            <p>Prediction: <span>${data.prediction}</span></p>
            <p>Confidence: <span>${data.confidence}</span></p>
        `;
    })
    .catch(err => {
        resultDiv.innerHTML = `<p style="color:red;">Error: ${err}</p>`;
    });
}
