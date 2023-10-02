
const mediadevice = navigator.mediaDevices;
const video = document.getElementById('video');
const show = document.getElementById('show');

show.addEventListener('click', () => {
   mediadevice.getUserMedia({
      video: true,
      audio: false,
   }).then((stream) => {
      video.srcObject = stream;

      video.addEventListener("loadedmetadata", () => {
         video.play();
      });
   }).catch(alert)
})