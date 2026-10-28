import {TraceClient, type TraceJson, type EndpointDecision} from './index.js';
const traces = new TraceClient({collection: 'support', source: 'app', onError: error => console.error(error.name)});
const result: number = await traces.decision({task: 'route', inputs: {message: 'Ticket'}}, trace => {
  const answer: TraceJson = {team: 'billing'};
  trace.setAnswer(answer);
  return 42;
});
const native: EndpointDecision = await traces.invokeEndpoint('support-triage', {state: {message: 'Ticket'}}, {version: 3});
await traces.reportBusinessOutcome(native.decision_id, {eventId: 'resolved1', values: {resolved: result === 42}});
await traces.reportOutcome(native.decision_id, {eventId: 'resolved1', values: {resolved: result === 42}});
await traces.close();
// @ts-expect-error non-JSON inputs are rejected
traces.decision({task: 'route', inputs: {callback: () => 1}}, () => {});
