function startVoiceSearch() {
    if (!('webkitSpeechRecognition' in window)) {
      alert("Your browser does not support speech recognition.");
      return;
    }
  
    const recognition = new webkitSpeechRecognition();
    recognition.lang = "en-US";
    recognition.start();
  
    recognition.onresult = function (event) {
      const transcript = event.results[0][0].transcript;
      document.getElementById("searchInput").value = transcript;
      searchMovie();
    };
  
    recognition.onerror = function (event) {
      alert("Voice recognition error: " + event.error);
    };
  }

function searchMovie() {
  const query = document.getElementById("searchInput").value.trim();
  const model = document.getElementById("modelSelector").value;

  // Hiển thị loading
  const resultsDiv = document.getElementById("results");
  // resultsDiv.innerHTML = "<p>Đang tìm...</p>";
  resultsDiv.innerHTML = `
    <p><strong>Đang tìm kiếm với model:</strong> ${model}</p>
  `;

  if (!query) {
    alert("Please enter a movie description or keyword.");
    return;
  }

  fetch("/search", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query, model }),
  })
    .then((res) => res.json())
    .then((data) => {
      console.log("Dữ liệu trả về:", data);
      displayMovies(data, model); 
    })
    .catch((error) => {
      console.error("Error fetching movies:", error);
      document.getElementById("results").innerHTML = "<p>Error fetching results.</p>";
    });
}

function displayMovies(movieList, model) {
  const resultsDiv = document.getElementById("results");
  resultsDiv.innerHTML =  ` `;
  // resultsDiv.innerHTML =  `<p><strong>Kết quả tìm kiếm với model:</strong> ${model}</p> <br>`;

  if (!movieList || movieList.length === 0) {
    resultsDiv.innerHTML = "<p>No movies found.</p>";
    return;
  }

  movieList.forEach((movie) => {
    const metadata = movie.payload?.metadata || {};

    const title = metadata.film_name || "Unknown Title";
    const runtime_minutes = metadata.runtime_minutes + " minutes"|| "N/A";
    const start_year = metadata.start_year || "N/A";
    const director = metadata.directors || "N/A";
    const genre = metadata.genres || "N/A";
    const description = movie.payload?.text || "";
    const image = metadata.image_link || "";
    const rating = metadata.rating || "N/A";
    const votes = metadata.votes || "N/A";
    const writers = metadata.writers || "N/A";

    const movieCard = document.createElement("div");
    movieCard.className = "movie-card";
    
    movieCard.innerHTML = `
    <div class="movie-info">
      <div class="left-image">
        ${image ? `<img src="${image}" alt="${title}" style="max-width:90%; max-height: 90%" />` : ""}
      </div>
      <div class="right-table">
        <table>
          <thead>
            <tr><h3><strong>${title}</h3></strong></tr>
          </thead>
          <tbody>
            <tr><td><strong>Director:</strong> </td><td> ${director}</td></tr>
            <tr><td><strong>Writers:</strong> </td><td> ${writers}</td></tr>
            <tr><td><strong>Duration:</strong> </td><td> ${runtime_minutes} </td></tr>
            <tr><td><strong>Genre:</strong> </td><td>${genre}</td></tr>
            <tr><td><strong>Start year:</strong> </td><td>${start_year}</td></tr>
            <tr><td><strong>Rating:</strong> </td><td>${rating}</td></tr>
            <tr><td><strong>Votes:</strong> </td><td>${votes}</td></tr>
            <tr><td><strong>Description:</strong> </td><td>${description}</td></tr>
          </tbody>
        </table>
      </div>
    </div>
    `;

    resultsDiv.appendChild(movieCard);
  });
}
document.getElementById("searchForm").addEventListener("submit", function (event) {
  event.preventDefault(); // Ngăn reload page
  searchMovie();          // Gọi hàm tìm kiếm
});
