import io
import json
import unittest
from unittest.mock import Mock
from saina.client import SainaHelm, SainaHelmError


class ClientTests(unittest.TestCase):
    def test_ask_memberships(self):
        c=SainaHelm('https://example.com',api_key='test')
        c._opener=Mock()
        request=dict(model='saina-helm-0.8b',state='text',questions={'tags':{'type':'multi_choice','question':'Tags?','options':{'a':None,'b':None}}})
        result={'model':'saina-helm-0.8b','answers':{'tags':{'type':'multi_choice','memberships':{'a':.9,'b':.8}}},'usage':{'input_tokens':1,'output_tokens':0}}
        def reply(req, **kwargs):
            self.assertEqual(req.full_url,'https://example.com/v1/ask')
            self.assertEqual(json.loads(req.data),dict(**request,mode='distribution'))
            return io.BytesIO(json.dumps(result).encode())
        c._opener.open.side_effect=reply
        self.assertEqual(c.ask(**request),result)
        for answers in [{}, {'tags':{'type':'yes_no','yes':.9,'no':.1}}, {'tags':{'type':'multi_choice','memberships':{'a':float('nan'),'b':.8}}}]:
            result['answers']=answers
            with self.assertRaises(SainaHelmError): c.ask(**request)

    def test_decision_request(self):
        c=SainaHelm('https://example.com',api_key='test')
        c._opener=Mock()
        result={'model':'saina-helm-0.8b','answers':{'q':{'type':'yes_no','yes':.1,'no':.9,'confidence':.8,'selected':False,'reason':'accepted'}}}
        c._opener.open.side_effect=lambda *a,**k:io.BytesIO(json.dumps(result).encode())
        c.ask(model='saina-helm-0.8b',state='text',mode='decision',threshold=.85,questions={'q':{'type':'yes_no','question':'Yes?'}})
        self.assertEqual(json.loads(c._opener.open.call_args.args[0].data)['threshold'],.85)

    def test_url(self):
        with self.assertRaises(ValueError): SainaHelm('http://example.com',api_key='test')

if __name__=='__main__': unittest.main()
