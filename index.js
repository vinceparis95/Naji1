import 'bootstrap/dist/css/bootstrap.css';
import * as tf from '@tensorflow/tfjs';

document.getElementById('output').innerText = "set plot";
//document.getElementById()
var input
input = prompt("what is your input?")
var n = [input]
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
const input = tf.tensor2d([4, 5, 3, 4], [4, 1]);
const ys = tf.tensor2d([40, 50, 30, 40], [4, 1]);
model.fit(input, ys, {epochs: 500}).then(() => {
    // Use model to predict values
    model.predict(tf.tensor2d(n, [1,1])).print();
});

