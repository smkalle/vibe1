// Record a reference run of the original's actor-critic (fly-matrix/web/learner.js on
// web/brain.js, which upstream holds to the Cadence library) for flysaver's learner parity test.
//
//   node tools/record_learner.mjs path/to/cadence-examples/fly-matrix/web > tests/learner_cases.txt
//
// Ten lessons with the page's configuration: settle 40 steps under an odour, decide (act with a
// recorded uniform draw), settle 5 more steps, then the outcome (learn, done). Sugar is on the
// banana (decaying fruit): approaching it pays 1, the bread 0, avoiding 0, and lesson 6 is a blow (-1).
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const web = process.argv[2];
const { SettlingBrain } = await import(pathToFileURL(join(web, "brain.js")));
const { ActorCriticLearner } = await import(pathToFileURL(join(web, "learner.js")));
globalThis.atob ??= (b) => Buffer.from(b, "base64").toString("binary");
const brain = new SettlingBrain(JSON.parse(readFileSync(join(web, "data", "brain.json"), "utf8")));
const config = { outputs: ["mbon:MBON11:right", "mbon:MBON05:left"], actions: [0, 1], plastic: { pre: ["kc"], post: ["mbon"] },
  critic: "kc", beta: 0.1, temperature: 0.3, nudgedSteps: 10, tolerance: 1e-3, gamma: 0.95, lam: 0.9, eta: 1.0, etaBias: 0,
  etaCritic: 0.05, cap: 3, dopamineCap: 1, tonic: {} };
const learner = new ActorCriticLearner(brain, config);
let seed = 7;
const uniform = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
const odours = { banana: ["orn:decaying_fruit:left", "orn:decaying_fruit:right"], bread: ["orn:yeasty:left", "orn:yeasty:right"] };
const sum = (a) => { let t = 0; for (const x of a) t += x; return t; };
console.log(`# reference: fly-matrix web/learner.js on web/brain.js, config ${JSON.stringify(config)}`);
console.log(`plastic ${learner.edges.length} critic ${learner.criticIndex.length}`);
for (let t = 0; t < 10; t++) {
  const fruit = t % 2 === 0 ? "banana" : "bread";
  brain.clearStimuli();
  for (const name of odours[fruit]) brain.stimulate(name, 0.8);
  brain.stimulate("haltere:left", 0.5); brain.stimulate("haltere:right", 0.5);
  for (let k = 0; k < 40; k++) brain.step();
  const u = uniform();
  const d = learner.act(false, u);
  for (let k = 0; k < 5; k++) brain.step();
  const reward = t === 6 ? -1 : d.choice === 0 && fruit === "banana" ? 1 : 0;
  const l = learner.learn(reward, true);
  console.log(`lesson ${fruit} ${u} ${reward} | ${d.p[0]} ${d.choice} ${d.value} | ${l.tdError} ${l.delta} ${l.moved} ${l.changed} ${l.meanAbsChange} ${sum(learner.efficacy)} ${learner.value()} ${sum(brain.s)}`);
}
