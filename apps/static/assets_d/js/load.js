// Function to start the logo animation when the page is loading
function startLogoAnimation() {
    // Get the logo element
    const logo = document.getElementById('logo');

    // Add a class to trigger the animation
    logo.classList.add('animate-logo');
}

// Add the event listener to start the animation when the page loads
window.onload = function () {
    startLogoAnimation();

    // Set a timeout to remove the logo element after the animation duration
    setTimeout(function() {
        const logo = document.getElementById('logo');
        logo.remove();
    }, 2000); // 2000 milliseconds (2 seconds) is the duration of the animation
};
