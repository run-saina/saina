# @saina-run/sdk

Pre-release Saina HTTP client. Not yet published to npm.

```js
import { Saina } from '@saina-run/sdk';

const client = new Saina({
  baseUrl: 'https://your-endpoint.example',
  apiKey: process.env.SAINA_API_KEY,
});

const result = await client.ask({
  model: 'helm-0.8b',
  state: 'I was charged twice.',
  questions: {
    issue: {
      type: 'single_choice',
      question: 'Identify the banking issue.',
      options: {
        duplicate: 'duplicate charge',
        delivery: 'card delivery',
      },
    },
  },
});
```

This client calls an existing endpoint; it does not run weights in JavaScript.
Apache-2.0. Saina is operated by Rama Labs Inc.
