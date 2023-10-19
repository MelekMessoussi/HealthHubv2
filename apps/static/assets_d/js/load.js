document.addEventListener("DOMContentLoaded", function () {
    // Show the spinner when the page starts loading
    document.getElementById("logo").style.display = "block";

    // You can also use AJAX or other methods to detect when the request is complete
    window.addEventListener("load", function () {
        // Hide the spinner when the page is fully loaded
        document.getElementById("logo").style.display = "none";
    });
});