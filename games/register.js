if('serviceWorker' in navigator){addEventListener('load',function(){navigator.serviceWorker.register('/games/sw.js',{scope:'/games/'}).catch(function(){});});}
