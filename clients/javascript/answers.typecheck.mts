import {Saina, type Selection, type Selections} from './index.js';

const saina = new Saina({apiKey: 'key'});
const answers = await saina.ask({
  model: 'saina-helm-2-0.8b',
  state: 'A ticket',
  questions: {toObject: {type: 'yes_no', question: 'Is a refund requested?'}},
});
const selections: Selections = answers.select({threshold: .9});
const selected: Selection | undefined = selections.get('toObject');
const serialized: Record<string, import('./index.js').Json> = selections.toObject();
const json: string = JSON.stringify(selections);
void [selected, serialized, json];
// @ts-expect-error question IDs must be strings
selections.get(1);
