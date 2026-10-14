"""Local-only segment listening review with persistent, private human answers."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_DATA=ROOT/'data/private/segment_pregrade_20260930'
TEMPLATE=ROOT/'scripts/acoustic_redo/templates/segment_pregrade_review_20260930.html'
# Policy for new writes only. Existing answer files, including legacy "unsure"
# values and their first-review snapshots, are loaded and exported unchanged.
JUDGEMENTS={'good','bad'}
SID=re.compile(r'^[A-Za-z0-9_-]+$')


def web_payload(payload):
    fields=['segment_id','sample_id','start_s','end_s','duration_s','phonation_s','score','state','score_sd',
            'n_models','explanations','audio_original','audio_denoised','waveform','events','audit_rank']
    return {'metadata':payload['metadata'],'recordings':payload['recordings'],
            'items':[{k:r[k] for k in fields if k in r} for r in payload['items']]}


def html_document(template,payload):
    if template.count('__DATA_JSON__')!=1:raise ValueError('template requires one data marker')
    encoded=json.dumps(web_payload(payload),ensure_ascii=False,allow_nan=False)
    encoded=encoded.replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    return template.replace('__DATA_JSON__',encoded).encode('utf-8')


class ReviewServer(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address,data_root=DEFAULT_DATA,audio_root=None,template_path=TEMPLATE):
        self.data_root=Path(data_root).resolve()
        self.audio_root=Path(audio_root or ROOT/'data/private/asr_refresh_20260930').resolve()
        self.payload=json.loads((self.data_root/'pregrades.json').read_text(encoding='utf-8'))
        self.items={r['segment_id']:r for r in self.payload['items']}
        self.sample_ids={r['sample_id'] for r in self.payload['items']}
        if len(self.items)!=len(self.payload['items']):raise ValueError('duplicate segment identity')
        self.html=html_document(Path(template_path).read_text(encoding='utf-8'),self.payload)
        self.lock=threading.RLock()
        self.answers_path=self.data_root/'human_answers.json'
        self.signature=self.payload['metadata']['audio_dataset_signature']
        self.state={'schema_version':1,'study':self.payload['metadata']['study'],
                    'audio_dataset_signature':self.signature,'answers':{},'exposure':{'segments':{},'recordings':{}}}
        if self.answers_path.exists():
            self.state=json.loads(self.answers_path.read_text(encoding='utf-8'))
            if self.state.get('audio_dataset_signature')!=self.signature:
                raise ValueError('existing human answers belong to different audio windows; preserve them before proceeding')
            if set(self.state.get('answers',{}))-set(self.items):raise ValueError('unknown stored segment answers')
        self.state.setdefault('exposure',{'segments':{},'recordings':{}})
        super().__init__(address,Handler)

    def persist(self,updated):
        temp=self.answers_path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(updated,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        temp.replace(self.answers_path);self.state=updated

    def save_exposure(self,data):
        if not isinstance(data,dict) or set(data)-{'segment_id','recording_ids','segment_ids'}:
            raise ValueError('invalid exposure fields')
        key=data.get('segment_id');recordings=data.get('recording_ids',[])
        visible=data.get('segment_ids',[])
        if key not in self.items or not isinstance(recordings,list) or any(not isinstance(s,str) or s not in self.sample_ids for s in recordings):
            raise ValueError('invalid exposure identity')
        if not isinstance(visible,list) or any(not isinstance(s,str) or s not in self.items for s in visible):raise ValueError('unknown exposed segment')
        stamp={'first_seen_utc':datetime.now(timezone.utc).isoformat(),'machine_version':self.payload['metadata']['version']}
        with self.lock:
            current=self.state['exposure'];segments=dict(current.get('segments',{}));records=dict(current.get('recordings',{}))
            segments.setdefault(key,stamp)
            for segment_id in visible:segments.setdefault(segment_id,stamp)
            for sid in recordings:records.setdefault(sid,stamp)
            exposure={'segments':segments,'recordings':records}
            self.persist({**self.state,'exposure':exposure})
        return {'saved':True,'exposure':exposure}

    def save_answer(self,data):
        if not isinstance(data,dict) or set(data)-{'segment_id','judgement','note','audio_source'}:
            raise ValueError('invalid answer fields')
        key=data.get('segment_id');judgement=data.get('judgement');note=data.get('note','')
        source=data.get('audio_source','original')
        if key not in self.items or judgement not in JUDGEMENTS or source not in {'original','denoised'} or not isinstance(note,str) or len(note)>5000:
            raise ValueError('invalid segment, judgement or note')
        item=self.items[key]
        with self.lock:
            seen=key in self.state['exposure']['segments'] or item['sample_id'] in self.state['exposure']['recordings']
            old=self.state['answers'].get(key)
            unchanged=old and old.get('judgement')==judgement and old.get('audio_source','original')==source
            answer={'judgement':judgement,'note':note,'updated_utc':datetime.now(timezone.utc).isoformat(),
                    'audio_source':source,'machine_score_at_review':item['score'],'machine_state_at_review':item['state'],
                    'machine_version_at_review':self.payload['metadata']['version'],
                    'machine_seen_before_last_save':seen,
                    'first_judgement':old.get('first_judgement',old['judgement']) if old else judgement,
                    'first_label_blind':old.get('first_label_blind') if old else not seen,
                    'first_audio_source':old.get('first_audio_source',old.get('audio_source')) if old else source,
                    'first_machine_score':old.get('first_machine_score',old.get('machine_score_at_review')) if old else item['score'],
                    'first_machine_state':old.get('first_machine_state',old.get('machine_state_at_review')) if old else item['state'],
                    'latest_label_blind':old.get('latest_label_blind') if unchanged else not seen}
            updated={**self.state,'answers':{**self.state['answers'],key:answer}}
            self.persist(updated)
        return {'saved':True,'segment_id':key,'answer':answer}


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass

    def allowed(self):
        port=self.server.server_address[1]
        host=self.headers.get('Host','')
        if host not in {f'127.0.0.1:{port}',f'localhost:{port}'}:
            self.send_error(403,'Local host only');return False
        return True

    def headers_common(self):
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Cache-Control','no-store')
        self.send_header('Referrer-Policy','no-referrer')

    def respond(self,body,content_type='application/json; charset=utf-8',head=False,download=None):
        if not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False,allow_nan=False).encode('utf-8')
        self.send_response(200);self.headers_common()
        self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(body)))
        if download:self.send_header('Content-Disposition',f'attachment; filename="{download}"')
        self.end_headers()
        if not head:self.wfile.write(body)

    def do_HEAD(self):self.get(head=True)
    def do_GET(self):self.get()

    def get(self,head=False):
        if not self.allowed():return
        path=urlparse(self.path).path
        if path in ('/','/index.html'):
            self.respond(self.server.html,'text/html; charset=utf-8',head);return
        if path in ('/api/state','/api/export'):
            with self.server.lock:state=json.loads(json.dumps(self.server.state))
            self.respond(state,head=head,download='segment_review_20260930.json' if path.endswith('export') else None);return
        match=re.fullmatch(r'/audio/(original|denoised)/([A-Za-z0-9_-]+)\.wav',path)
        if match:
            variant,sid=match.groups()
            if sid not in self.server.sample_ids:self.send_error(404);return
            audio=(self.server.audio_root/f'audio_{variant}'/f'{sid}.wav').resolve()
            if not audio.is_relative_to(self.server.audio_root) or not audio.is_file():self.send_error(404);return
            self.audio(audio,head);return
        if path=='/favicon.ico':self.send_response(204);self.end_headers();return
        self.send_error(404)

    def audio(self,path,head):
        size=path.stat().st_size;start,end=0,size-1;status=200
        raw=self.headers.get('Range')
        if raw:
            match=re.fullmatch(r'bytes=(\d*)-(\d*)',raw)
            if not match or not any(match.groups()):self.send_error(416);return
            a,b=match.groups()
            if not a:
                amount=int(b)
                if amount<=0:self.send_error(416);return
                start=max(0,size-amount)
            else:
                start=int(a);end=min(int(b),size-1) if b else size-1
            if start>=size or end<start:
                self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.end_headers();return
            status=206
        self.send_response(status);self.headers_common()
        self.send_header('Content-Type','audio/wav');self.send_header('Accept-Ranges','bytes')
        self.send_header('Content-Length',str(end-start+1))
        if status==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers()
        if head:return
        try:
            with path.open('rb') as f:
                f.seek(start);remaining=end-start+1
                while remaining:
                    chunk=f.read(min(65536,remaining))
                    if not chunk:break
                    self.wfile.write(chunk);remaining-=len(chunk)
        except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass

    def do_POST(self):
        if not self.allowed():return
        path=urlparse(self.path).path
        if path not in {'/api/answer','/api/exposure'}:self.send_error(404);return
        port=self.server.server_address[1]
        origin=self.headers.get('Origin')
        if origin and origin not in {f'http://127.0.0.1:{port}',f'http://localhost:{port}'}:
            self.send_error(403,'Same local origin required');return
        if self.headers.get('Content-Type','').split(';')[0]!='application/json':self.send_error(415);return
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=30000:raise ValueError('answer size invalid')
            data=json.loads(self.rfile.read(size).decode('utf-8'))
            result=self.server.save_answer(data) if path=='/api/answer' else self.server.save_exposure(data)
        except (ValueError,TypeError,KeyError,json.JSONDecodeError):
            self.send_error(400,'Invalid answer');return
        except OSError:
            self.send_error(500,'Answer could not be saved');return
        self.respond(result)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8874)
    parser.add_argument('--data-root',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--audio-root',type=Path)
    parser.add_argument('--template',type=Path,default=TEMPLATE)
    args=parser.parse_args()
    server=ReviewServer(('127.0.0.1',args.port),args.data_root,args.audio_root,args.template)
    print(f'Listening review ready: http://127.0.0.1:{server.server_address[1]}/',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
