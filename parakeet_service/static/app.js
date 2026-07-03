(() => {
  const TARGET_SR = 16000;
  const el = (id) => document.getElementById(id);
  const app = el('app'), statusEl = el('status'), micBtn = el('micBtn');
  const flow = el('flow'), caret = el('caret'), transcript = el('transcript');
  const mStatus = el('mStatus'), mLat = el('mLat'), mConf = el('mConf'),
        mChunks = el('mChunks'), mSr = el('mSr'), lvlEl = el('lvl');
  const micK = el('micK'), micH = el('micH');

  let ws=null, audioCtx=null, stream=null, source=null, processor=null, analyser=null, sink=null;
  let running=false, chunks=0, queuedTs=[], raf=0, confSum=0, confN=0;

  function setStatus(s, label){ statusEl.dataset.s=s; statusEl.querySelector('.txt').textContent=label; mStatus.textContent=label; }

  // ---------- audio helpers ----------
  function downsample(buf, inRate, outRate){
    if(outRate===inRate) return buf;
    const ratio=inRate/outRate, len=Math.floor(buf.length/ratio), out=new Float32Array(len);
    let pos=0;
    for(let i=0;i<len;i++){
      const next=Math.floor((i+1)*ratio); let sum=0,c=0;
      for(let j=Math.floor(i*ratio); j<next && j<buf.length; j++){ sum+=buf[j]; c++; }
      out[i]= c? sum/c : buf[pos]; pos=next;
    }
    return out;
  }
  function floatTo16(buf){
    const out=new Int16Array(buf.length);
    for(let i=0;i<buf.length;i++){ let s=Math.max(-1,Math.min(1,buf[i])); out[i]= s<0 ? s*0x8000 : s*0x7fff; }
    return out;
  }

  // ---------- waveform ----------
  const cv=el('scope'), ctx2=cv.getContext('2d'); let W=0,H=0,dpr=1;
  function resize(){ dpr=Math.min(2,window.devicePixelRatio||1); W=cv.clientWidth; H=cv.clientHeight; cv.width=W*dpr; cv.height=H*dpr; ctx2.setTransform(dpr,0,0,dpr,0,0); }
  window.addEventListener('resize', resize);
  let idlePhase=0;
  function draw(){
    raf=requestAnimationFrame(draw);
    ctx2.clearRect(0,0,W,H);
    const mid=H/2;
    ctx2.lineWidth=1.6; ctx2.lineJoin='round';
    const grad=ctx2.createLinearGradient(0,0,W,0);
    grad.addColorStop(0,'rgba(246,167,35,.25)'); grad.addColorStop(.5,'#f6a723'); grad.addColorStop(1,'rgba(246,167,35,.25)');
    ctx2.strokeStyle=grad; ctx2.shadowColor='rgba(246,167,35,.5)'; ctx2.shadowBlur=10;
    ctx2.beginPath();
    if(analyser && running){
      const n=analyser.fftSize, data=new Uint8Array(n); analyser.getByteTimeDomainData(data);
      let peak=0;
      for(let i=0;i<W;i++){ const s=data[Math.floor(i/W*n)]/128-1; if(Math.abs(s)>peak)peak=Math.abs(s); const y=mid+s*mid*0.92; i?ctx2.lineTo(i,y):ctx2.moveTo(i,y); }
      const db = peak>0.0005 ? (20*Math.log10(peak)).toFixed(0) : '-inf';
      lvlEl.textContent = db+' dB';
    } else {
      idlePhase+=0.03;
      for(let i=0;i<W;i++){ const y=mid + Math.sin(i*0.03+idlePhase)*1.2 + Math.sin(i*0.11+idlePhase*1.7)*0.6; i?ctx2.lineTo(i,y):ctx2.moveTo(i,y); }
    }
    ctx2.stroke(); ctx2.shadowBlur=0;
  }

  // ---------- transcript ----------
  function addUtterance(text, conf, words){
    if(!text || !text.trim()) return;
    document.querySelectorAll('.utt').forEach(u=>u.classList.add('old'));
    const span=document.createElement('span'); span.className='utt';
    // low-confidence hinting when per-word scores are available
    if(Array.isArray(words) && words.length){
      const toks=text.trim().split(/\s+/);
      toks.forEach((t,i)=>{
        const w=document.createElement('span');
        if(words[i]!=null && words[i]<0.55) w.className='lowconf';
        w.textContent=t+' '; span.appendChild(w);
      });
    } else { span.textContent=text.trim()+' '; }
    flow.insertBefore(span, caret);
    transcript.classList.add('has-text');
    transcript.scrollTop = transcript.scrollHeight;
    chunks++; mChunks.textContent=chunks;
    if(typeof conf==='number'){ confSum+=conf; confN++; mConf.textContent=Math.round(conf*100); }
    const t0=queuedTs.shift();
    if(t0) mLat.textContent = Math.round(performance.now()-t0);
  }

  // ---------- websocket + capture ----------
  function wsURL(){
    const proto = location.protocol==='https:'?'wss:':'ws:';
    const params = new URLSearchParams(location.search);
    const key = params.get('api_key');
    return proto+'//'+location.host+'/ws'+(key?('?api_key='+encodeURIComponent(key)):'');
  }

  async function start(){
    if(!window.isSecureContext || !(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)){
      showInsecure(); return;
    }
    try{
      setStatus('connecting','Connecting');
      stream = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true,autoGainControl:true}});
      audioCtx = new (window.AudioContext||window.webkitAudioContext)({sampleRate:TARGET_SR});
      if(audioCtx.state==='suspended') await audioCtx.resume();
      mSr.textContent = Math.round(audioCtx.sampleRate/1000);
      source = audioCtx.createMediaStreamSource(stream);
      analyser = audioCtx.createAnalyser(); analyser.fftSize=1024; analyser.smoothingTimeConstant=.6;
      source.connect(analyser);
      processor = audioCtx.createScriptProcessor(4096,1,1);
      sink = audioCtx.createGain(); sink.gain.value=0;
      source.connect(processor); processor.connect(sink); sink.connect(audioCtx.destination);

      ws = new WebSocket(wsURL()); ws.binaryType='arraybuffer';
      ws.onopen = () => { running=true; app.classList.add('rec'); flow.classList.add('live');
        setStatus('live','Listening'); micK.textContent='Stop'; micH.textContent='Streaming audio';
        el('emptyBig').textContent='Listening…'; el('emptySmall').textContent='Speak — words stream in live'; };
      ws.onmessage = (ev) => {
        let m; try{ m=JSON.parse(ev.data);}catch{ return; }
        if(m.status==='queued'){ queuedTs.push(performance.now()); return; }
        if('text' in m) addUtterance(m.text, m.confidence, m.words);
      };
      ws.onerror = () => { setStatus('error','Socket error'); };
      ws.onclose = () => { if(running) stop(); };

      processor.onaudioprocess = (e) => {
        if(!running || !ws || ws.readyState!==WebSocket.OPEN) return;
        const input=e.inputBuffer.getChannelData(0);
        const ds=downsample(input, audioCtx.sampleRate, TARGET_SR);
        ws.send(floatTo16(ds).buffer);
      };
    }catch(err){
      setStatus('error','Mic denied');
      showBanner('Microphone unavailable', [
        'Access was blocked or no input device was found. Check the browser permission prompt and your system microphone.'
        + (err && err.message ? ' (' + err.message + ')' : '')
      ]);
      cleanup();
    }
  }

  function stop(){
    running=false; app.classList.remove('rec'); flow.classList.remove('live');
    setStatus('idle','Standby'); micK.textContent='Listen'; micH.textContent='Tap or press Space';
    el('emptyBig').textContent='“Press listen, then speak.”'; el('emptySmall').textContent='Words appear here in real time';
    try{ ws&&ws.readyState<=1&&ws.close(); }catch{}
    cleanup();
  }
  function cleanup(){
    try{ processor&&(processor.onaudioprocess=null); }catch{}
    try{ stream&&stream.getTracks().forEach(t=>t.stop()); }catch{}
    try{ audioCtx&&audioCtx.close(); }catch{}
    ws=null; processor=null; analyser=null; audioCtx=null; stream=null; queuedTs=[];
    lvlEl.textContent='-inf dB';
  }

  function toggle(){ running ? stop() : start(); }

  // ---------- banners (built as DOM nodes, no innerHTML) ----------
  // parts: array of strings (text) or {code:'...'} chips.
  function showBanner(title, parts){
    el('bannerT').textContent = title;
    const d = el('bannerD'); d.textContent = '';
    parts.forEach(p => {
      if (typeof p === 'string') { d.appendChild(document.createTextNode(p)); }
      else if (p && p.code) { const c = document.createElement('code'); c.textContent = p.code; d.appendChild(c); }
    });
    el('banner').classList.add('show');
  }
  function showInsecure(){
    showBanner('Secure context required for microphone', [
      'Browsers only allow mic capture over ', {code:'https://'}, ' or on ', {code:'localhost'},
      '. You are on ', {code: location.protocol + '//' + location.host},
      '. Open this page over HTTPS (accept the self-signed certificate), or from the server itself at ',
      {code:'http://localhost:8000'}, '.'
    ]);
    setStatus('error','Needs HTTPS');
  }
  el('bannerX').onclick=()=>el('banner').classList.remove('show');

  micBtn.addEventListener('click', toggle);
  document.addEventListener('keydown', (e)=>{ if(e.code==='Space' && e.target===document.body){ e.preventDefault(); toggle(); } });

  resize(); draw();
  if(!window.isSecureContext) showInsecure();
})();
