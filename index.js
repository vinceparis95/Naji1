import 'bootstrap/dist/css/bootstrap.css';
import * as tf from '@tensorflow/tfjs';


let userInput = prompt("enter number")
let n = [userInput]
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
const input = tf.tensor2d([4, 5, 3, 4], [4, 1]);
const ys = tf.tensor2d([20, 25, 15, 20], [4, 1]);
model.fit(input, ys, {epochs: 500}).then(() => {
    model.predict(tf.tensor2d(n, [1, 1])).print();
})

