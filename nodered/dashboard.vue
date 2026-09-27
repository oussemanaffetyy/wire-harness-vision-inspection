<template>
  <main ref="screen" class="whi-screen">
    <header class="whi-header">
      <div><p class="whi-eyebrow">POSTE 01 · DASHBOARD LEONI</p><h1>Contrôle du faisceau<span>.</span></h1></div>
    </header>
    <div class="whi-layout">
      <section class="whi-panel whi-camera" aria-label="Vidéo d'inspection">
        <header class="whi-panel-header"><h2>Vue du poste</h2><div class="whi-camera-actions"><button class="whi-fullscreen" :aria-label="fullscreen ? 'Quitter le plein écran' : 'Ouvrir en plein écran'" :title="fullscreen ? 'Quitter le plein écran' : 'Plein écran'" @click="toggleFullscreen"><svg v-if="!fullscreen" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5"/></svg><svg v-else viewBox="0 0 24 24" aria-hidden="true"><path d="M9 4v5H4M20 9h-5V4M15 20v-5h5M4 15h5v5"/></svg></button></div></header>
        <div class="whi-image-area">
          <img v-if="video.image_base64" :src="'data:image/jpeg;base64,' + video.image_base64" alt="Vue du poste avec les masques de segmentation YOLO" />
          <div v-else class="whi-empty-video"><span class="whi-viewfinder" aria-hidden="true"></span></div>
          <span v-if="video.image_base64 && !videoLive" class="whi-paused-image">Dernière image reçue · vidéo en pause</span>
        </div>
        <footer class="whi-legend"><span class="whi-connector">Connecteur</span><span class="whi-clip">Clip</span><span class="whi-cable">Câble</span></footer>
      </section>
      <div class="whi-side">
        <section class="whi-panel whi-verdict" :class="verdictClass" aria-label="Résultat de l'inspection">
          <div class="whi-verdict-top"><h2>Résultat de l'inspection</h2><time>{{ current ? time(current.timestamp) : '—' }}</time></div>
          <div class="whi-verdict-main" aria-live="polite"><strong>{{ displayStatus }}</strong></div>
        </section>

        <section class="whi-panel whi-history" aria-label="Historique des inspections">
          <header class="whi-panel-header"><h2>Historique récent</h2><button v-if="journalName" class="whi-link" @click="closeJournal">Session</button></header>
          <div class="whi-history-tools">
            <div v-if="filtered.length" class="whi-filters" aria-label="Filtrer l'historique"><button :aria-pressed="filter === 'all'" @click="setFilter('all')">Tout</button><button :aria-pressed="filter === 'NOK'" @click="setFilter('NOK')">Défauts</button></div>
            <div class="whi-file-actions whi-secondary-actions"><button @click="download" :disabled="!filtered.length" title="Télécharger l’historique">Exporter TXT</button></div>
          </div>
          <p v-if="notice" class="whi-notice" role="status">{{ notice }}</p>
          <ol v-if="visibleRows.length" class="whi-rows">
            <li v-for="row in visibleRows" :key="row.id"><span class="whi-badge" :class="row.status === 'NOK' ? 'is-nok' : 'is-ok'">{{ row.status }}</span><div class="whi-row-description"><p :title="row.detail">{{ describe(row) }}</p><small>{{ basename(row.source_name) || 'Journal local' }}</small></div><time :title="date(row.timestamp)">{{ time(row.timestamp) }}<small>{{ day(row.timestamp) }}</small></time></li>
          </ol>
          <div v-else class="whi-empty-history"><span class="whi-history-empty-mark">—</span><h3>{{ filter === 'NOK' ? 'Aucun défaut' : 'Aucun résultat récent' }}</h3></div>
          <footer v-if="pageCount > 1" class="whi-history-footer"><div><button aria-label="Page précédente" :disabled="safePage === 0" @click="page = safePage - 1">‹</button><span>{{ safePage + 1 }} / {{ pageCount }}</span><button aria-label="Page suivante" :disabled="safePage >= pageCount - 1" @click="page = safePage + 1">›</button></div></footer>
        </section>
      </div>
    </div>
  </main>
</template>

<script>
export default {
  data () {
    return { current: null, video: {}, metrics: {}, history: [], imported: [], journalName: '', notice: '', loading: false,
      filter: 'all', page: 0, now: Date.now(), connected: false, timer: null, fullscreen: false }
  },
  computed: {
    live () { return this.connected && this.recent(this.current?.timestamp) },
    videoLive () { return this.connected && this.recent(this.video.timestamp) },
    verdictClass () { return this.live ? (this.current.status === 'NOK' ? 'is-nok' : 'is-ok') : 'is-idle' },
    displayStatus () { return this.live ? this.current.status : (this.current ? 'PAUSE' : '—') },
    filtered () { const rows = this.journalName ? this.imported : this.history; return this.filter === 'NOK' ? rows.filter(row => row.status === 'NOK') : rows },
    pageCount () { return Math.max(1, Math.ceil(this.filtered.length / 4)) },
    safePage () { return Math.min(this.page, this.pageCount - 1) },
    visibleRows () { return this.filtered.slice(this.safePage * 4, this.safePage * 4 + 4) }
  },
  watch: {
    msg: { immediate: true, handler (message) {
      const p = message?.payload
      if (!p || typeof p !== 'object') return
      if (p.kind === 'inspection') {
        this.current = p.current || null
        if (Array.isArray(p.history)) this.history = p.history.slice(0, 300)
      } else if (Number.isFinite(p.total_frames) || Number.isFinite(p.ok_count) || Number.isFinite(p.nok_count)) {
        this.metrics = { ...this.metrics, ...p }
      } else if (typeof p.image_base64 === 'string' && p.image_base64.length < 6000000) {
        this.video = p
      }
    } }
  },
  mounted () {
    this.connected = this.$socket.connected
    this.$socket.on('connect', this.onConnect)
    this.$socket.on('disconnect', this.onDisconnect)
    document.addEventListener('fullscreenchange', this.onFullscreenChange)
    this.send({ topic: 'inspection/request' })
    this.timer = setInterval(() => { this.now = Date.now() }, 1000)
  },
  unmounted () {
    clearInterval(this.timer)
    this.$socket.off('connect', this.onConnect)
    this.$socket.off('disconnect', this.onDisconnect)
    document.removeEventListener('fullscreenchange', this.onFullscreenChange)
  },
  methods: {
    onConnect () { this.connected = true; this.send({ topic: 'inspection/request' }) },
    onDisconnect () { this.connected = false },
    onFullscreenChange () { this.fullscreen = document.fullscreenElement === this.$refs.screen },
    async toggleFullscreen () {
      try {
        if (document.fullscreenElement === this.$refs.screen) await document.exitFullscreen()
        else await this.$refs.screen.requestFullscreen()
      } catch (error) {
        this.notice = 'Le plein écran n’est pas disponible dans ce navigateur.'
      }
    },
    recent (timestamp) { const age = this.now - Date.parse(timestamp); return Number.isFinite(age) && age >= -5000 && age < 15000 },
    number (value) { return Number.isFinite(Number(value)) ? Number(value).toLocaleString('fr-FR') : '—' },
    basename (value) { return typeof value === 'string' ? value.split(/[\\/]/).pop() : '' },
    time (value) { return new Date(value).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit', second: '2-digit' }) },
    day (value) { return new Date(value).toLocaleDateString('fr-FR', { day: '2-digit', month: '2-digit' }) },
    date (value) { return new Date(value).toLocaleString('fr-FR') },
    describe (row) {
      if (row.status === 'NOK') return /clip.*connect/i.test(row.detail) ? 'Clip au contact du connecteur' : (row.detail || 'Défaut signalé')
      if (/d.tection absente/i.test(row.detail)) return 'Aucun défaut confirmé · détection partielle'
      return row.detail && !/^(OK par d.faut|status=OK)/i.test(row.detail) ? row.detail : 'Aucun défaut confirmé'
    },
    setFilter (value) { this.filter = value; this.page = 0 },
    closeJournal () { this.journalName = ''; this.imported = []; this.notice = ''; this.page = 0 },
    parseJournal (text) {
      const rows = []
      let previous = null
      let skipped = 0
      let matched = 0
      for (const line of text.replace(/^\uFEFF/, '').split(/\r?\n/)) {
        if (!line.trim() || line.startsWith('#')) continue
        const match = line.match(/^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s*-\s*Test (OK|NOK)(?::\s*(.*))?\s*$/)
        if (!match || !Number.isFinite(Date.parse(match[1].replace(' ', 'T')))) { skipped++; continue }
        matched++
        const row = { timestamp: match[1].replace(' ', 'T'), status: match[2], detail: (match[3] || '').slice(0, 512), source_name: '', id: `txt-${matched}` }
        if (previous !== row.status) rows.push(row)
        previous = row.status
        if (rows.length > 300) rows.shift()
      }
      return { rows: rows.reverse(), skipped, matched }
    },
    async importJournal (event) {
      const file = event.target.files?.[0]
      event.target.value = ''
      if (!file) return
      if (file.size > 20 * 1024 * 1024) { this.notice = 'Fichier trop volumineux (20 Mo maximum). Ouvrez un extrait du journal.'; return }
      this.loading = true
      try {
        const parsed = this.parseJournal(await file.text())
        if (!parsed.matched) throw new Error('Aucune ligne reconnue. Choisissez le fichier IACom.txt généré par Python.')
        this.imported = parsed.rows
        this.journalName = file.name
        this.filter = 'all'
        this.page = 0
        this.notice = `Journal lu sur cet écran uniquement. Résultats consécutifs identiques regroupés.${parsed.skipped ? ' ' + parsed.skipped + ' ligne(s) ignorée(s).' : ''}`
      } catch (error) { this.notice = error.message || 'Lecture du fichier impossible.' }
      finally { this.loading = false }
    },
    localTimestamp (value) {
      const d = new Date(value)
      const pad = n => String(n).padStart(2, '0')
      return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
    },
    download () {
      const lines = ['# Historique des changements OK/NOK - pas un comptage de pieces', '# Horaires locaux du navigateur']
      for (const row of [...this.filtered].reverse()) {
        const detail = row.detail.replace(/[\r\n]+/g, ' ')
        lines.push(`[${this.localTimestamp(row.timestamp)}] - Test ${row.status}${detail ? ': ' + detail : ''}`)
      }
      const url = URL.createObjectURL(new Blob([lines.join('\r\n') + '\r\n'], { type: 'text/plain;charset=utf-8' }))
      const a = document.createElement('a')
      a.href = url
      a.download = `historique_inspection_${this.localTimestamp(Date.now()).replace(/[: ]/g, '-')}.txt`
      document.body.appendChild(a)
      a.click()
      a.remove()
      setTimeout(() => URL.revokeObjectURL(url), 1000)
    }
  }
}
</script>

<style>
.whi-dashboard-group, .whi-dashboard-widget { background: transparent !important; border: 0 !important; box-shadow: none !important; }
.whi-dashboard-group > .v-card-text { padding: 0 !important; }
.whi-dashboard-widget { height: auto !important; overflow: visible !important; }
.whi-screen { --whi-bg: #151c21; --whi-line: #34424c; --whi-ink: #eff3f4; --whi-muted: #acbac3; --whi-green: #8bd6b1; --whi-red: #ff9b8b; color: var(--whi-ink); font-family: 'Bahnschrift', 'Avenir Next', 'Trebuchet MS', sans-serif; width: 100%; height: 100vh; min-height: 0; overflow: hidden; display: flex; flex-direction: column; padding: 10px 12px 0; box-sizing: border-box; }
.whi-screen *, .whi-screen *::before, .whi-screen *::after { box-sizing: border-box; }
.whi-screen h1, .whi-screen h2, .whi-screen h3, .whi-screen p { margin: 0; }
.whi-header { display: flex; justify-content: space-between; align-items: center; gap: 20px; flex: 0 0 auto; margin-bottom: 16px; }
.whi-eyebrow { font-size: 12px; font-weight: 700; letter-spacing: 2.3px; color: #a8b8c0; margin-bottom: 7px !important; }
.whi-header h1 { font-weight: 600; font-size: clamp(26px, 3vw, 38px); letter-spacing: -1.2px; line-height: 1.2; }
.whi-header h1 span { color: #8bd6b1; }
.whi-layout { display: grid; grid-template-columns: minmax(0, 1.65fr) minmax(360px, 1fr); gap: 18px; flex: 1 1 auto; min-height: 0; align-items: stretch; }
.whi-panel { border: 1px solid var(--whi-line); background: var(--whi-bg); border-radius: 12px; min-width: 0; }
.whi-panel-header { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 18px 20px; }
.whi-camera-actions { display: flex; align-items: center; gap: 12px; min-width: 0; }
.whi-fullscreen { display: grid; place-items: center; width: 32px; height: 32px; padding: 6px !important; flex-shrink: 0; }
.whi-fullscreen svg { width: 18px; height: 18px; fill: none; stroke: currentColor; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }
.whi-screen h2 { font-size: 18px; font-weight: 600; }
.whi-panel-header p { color: var(--whi-muted); font-size: 14px; margin-top: 5px; overflow-wrap: anywhere; }
.whi-camera { display: flex; flex-direction: column; min-height: 0; overflow: hidden; }
.whi-image-area { position: relative; flex: 1 1 auto; min-height: 0; height: auto; background: #0e1419; display: grid; place-items: center; }
.whi-image-area img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }
.whi-empty-video { padding: 24px; text-align: center; }
.whi-viewfinder { display: block; width: 64px; height: 50px; border: 1px dashed #5c707e; border-radius: 8px; }
.whi-screen:fullscreen { width: 100vw; height: 100vh; min-height: 0; overflow: hidden; padding: 20px; background: #10171c; }
.whi-screen:fullscreen .whi-layout { grid-template-columns: minmax(0, 1.65fr) minmax(360px, 1fr); }
.whi-screen:fullscreen .whi-image-area { min-height: 0; }
.whi-paused-image { position: absolute; bottom: 14px; left: 14px; right: 14px; padding: 7px 12px; background: #111a20ed; color: #dae1e5; font-size: 11px; border-radius: 5px; text-align: center; }
.whi-legend { display: flex; flex-wrap: wrap; align-items: center; gap: 16px; padding: 13px 20px; border-top: 1px solid var(--whi-line); background: #121a1f; color: var(--whi-muted); font-size: 14px; }
.whi-legend span::before { content: ''; display: inline-block; width: 10px; height: 3px; margin: 0 7px 3px 0; background: var(--legend); }
.whi-connector { --legend: #ed706a; }.whi-clip { --legend: #6fafff; }.whi-cable { --legend: #73ce91; }
.whi-legend small { margin-left: auto; color: #8498a6; font-size: 10px; }
.whi-side { display: flex; flex-direction: column; gap: 12px; min-width: 0; min-height: 0; overflow: hidden; }
.whi-verdict { padding: 18px; border-top: 4px solid #657a88; }
.whi-verdict.is-ok { border-top-color: var(--whi-green); background: linear-gradient(115deg, #1c302b, var(--whi-bg) 75%); }
.whi-verdict.is-nok { border-top-color: var(--whi-red); background: linear-gradient(115deg, #362522, var(--whi-bg) 75%); }
.whi-verdict-top { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.whi-verdict-top time { font-size: 15px; font-variant-numeric: tabular-nums; color: var(--whi-muted); }
.whi-verdict-main { display: flex; align-items: center; justify-content: center; min-height: 190px; margin: 10px 0 4px; border-radius: 8px; background: rgba(120, 140, 150, .08); }
.whi-verdict.is-ok .whi-verdict-main { background: rgba(65, 160, 112, .2); }
.whi-verdict.is-nok .whi-verdict-main { background: rgba(205, 71, 63, .2); }
.whi-verdict-main strong { font-size: clamp(76px, 8vw, 120px); letter-spacing: -5px; line-height: 1; font-weight: 750; }
.is-ok .whi-verdict-main strong { color: var(--whi-green); }.is-nok .whi-verdict-main strong { color: var(--whi-red); }.is-idle .whi-verdict-main strong { font-size: 34px; color: #74838b; letter-spacing: 0; font-weight: 400; }
.whi-counters { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1px; padding: 0; overflow: hidden; }
.whi-counter { min-width: 0; padding: 13px 18px 15px; background: #192329; }
.whi-counter span { display: block; color: #c0cdd3; font-size: 11px; font-weight: 600; letter-spacing: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.whi-counter strong { display: block; margin-top: 4px; color: #f1f5f6; font-size: 32px; line-height: 1; font-variant-numeric: tabular-nums; }
.whi-counter.is-ok strong { color: var(--whi-green); }.whi-counter.is-nok strong { color: var(--whi-red); }
.whi-history { flex: 1 1 auto; min-height: 180px; display: flex; flex-direction: column; overflow-y: auto; }
.whi-screen button { font-family: inherit; font-size: 14px; line-height: 1.3; color: #dbe3e8; border: 1px solid #41515d; border-radius: 6px; padding: 9px 11px; background: transparent; cursor: pointer; transition: background .15s; }
.whi-screen button:hover:not(:disabled) { background: #33434d; }
.whi-screen button:focus-visible { outline: 2px solid #8bd6b1; outline-offset: 3px; }
.whi-screen button:disabled { opacity: .4; cursor: default; }
.whi-history-tools { display: flex; gap: 10px; justify-content: space-between; flex-wrap: wrap; padding: 0 20px 14px; }
.whi-filters, .whi-file-actions { display: flex; gap: 5px; }
.whi-secondary-actions button { color: #aebdc5; border-color: #34424c; font-size: 12px; padding: 6px 9px; }
.whi-filters { border: 1px solid var(--whi-line); padding: 3px; border-radius: 7px; }
.whi-filters button { border: 0; padding: 6px 10px; }
.whi-filters button[aria-pressed=true] { background: #34454f; color: #fff; }
.whi-screen .whi-link { border: 0; color: #b7d5e4; padding: 4px 0; text-align: right; }
.whi-file-input { display: none; }
.whi-notice { color: #b8d2df; background: #20313c; font-size: 14px; line-height: 1.5; padding: 10px 20px; }
.whi-rows { list-style: none; padding: 0 20px; margin: 0; overflow: visible; }
.whi-rows li { display: flex; align-items: center; gap: 12px; border-top: 1px solid var(--whi-line); padding: 13px 0; min-height: 66px; }
.whi-badge { display: inline-flex; width: 42px; height: 26px; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; letter-spacing: .4px; border-radius: 4px; flex-shrink: 0; }
.whi-badge.is-ok { background: #203c32; color: var(--whi-green); }.whi-badge.is-nok { background: #462d28; color: var(--whi-red); }
.whi-row-description { flex: 1; min-width: 0; }
.whi-row-description p { font-size: 15px; line-height: 1.4; overflow-wrap: anywhere; }
.whi-rows small { display: block; margin-top: 4px; font-size: 12px; color: #a5b6c1; overflow-wrap: anywhere; }
.whi-rows time { text-align: right; color: #dbe3e8; font-size: 14px; font-variant-numeric: tabular-nums; flex-shrink: 0; }
.whi-empty-history { padding: 18px 20px; color: var(--whi-muted); text-align: center; }
.whi-empty-history h3 { font-weight: 500; font-size: 14px; }
.whi-empty-history p { font-size: 14px; line-height: 1.6; }
.whi-history-footer { display: flex; justify-content: flex-end; align-items: center; gap: 10px; margin-top: auto; padding: 10px 20px; border-top: 1px solid var(--whi-line); font-size: 13px; color: var(--whi-muted); }
.whi-history-footer > div { display: flex; align-items: center; gap: 10px; }
.whi-history-footer button { font-size: 17px; padding: 2px 9px; }
@media (min-width: 1024px) { .whi-header { animation: whi-enter .3s ease-out; } .whi-layout { animation: whi-enter .45s ease-out; } }
@keyframes whi-enter { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: none; } }
@keyframes whi-spin { to { transform: rotate(360deg); } }
@media (max-width: 1000px) { .whi-layout { grid-template-columns: 1fr; } .whi-image-area { height: 480px; min-height: 0; flex: none; } .whi-side { display: grid; grid-template-columns: minmax(0, .8fr) minmax(0, 1.2fr); } .whi-verdict-main { min-height: 130px; } }
@media (max-width: 680px) { .whi-screen { padding: 4px 0 0; } .whi-header { margin-bottom: 20px; } .whi-side { display: flex; } .whi-image-area { height: 420px; } .whi-panel-header, .whi-verdict { padding: 16px; } .whi-history-tools { padding: 0 16px 12px; } .whi-rows { padding: 0 16px; } .whi-verdict-main { min-height: 112px; } .whi-legend { gap: 10px; padding: 13px 16px; } }
@media (prefers-reduced-motion: reduce) { .whi-screen *, .whi-header, .whi-layout { animation: none !important; transition: none !important; } }
</style>
