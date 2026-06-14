/* ============================================================
   NeoJaundice AI — Happy Baby Playground (POPOLATED V6)
   Developed by Rahul Patel (IIIT Surat)
   ============================================================ */

class Toy {
    constructor(w, h, x, y) {
        this.reset(w, h, x, y);
    }
    reset(w, h, x, y) {
        this.x = x || Math.random() * w;
        this.y = y || -50;
        this.r = Math.random() * 12 + 8;
        this.vx = (Math.random() - 0.5) * 5;
        this.vy = Math.random() * 2 + 1;
        this.color = ['#ff6b6b', '#ffd93d', '#6eccff', '#a8e6cf', '#ff8b94'][Math.floor(Math.random() * 5)];
        this.gravity = 0.18;
        this.bounce = 0.7;
        this.friction = 0.99;
    }
    update(w, h, mouse) {
        this.vy += this.gravity;
        this.x += this.vx;
        this.y += this.vy;
        this.vx *= this.friction;

        if (this.y + this.r > h) {
            this.y = h - this.r;
            this.vy *= -this.bounce;
        }
        if (this.x + this.r > w || this.x - this.r < 0) {
            this.vx *= -this.bounce;
            this.x = this.x < this.r ? this.r : w - this.r;
        }

        const dx = this.x - mouse.x;
        const dy = this.y - mouse.y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        if (dist < 100) {
            const angle = Math.atan2(dy, dx);
            const force = (100 - dist) * 0.2;
            this.vx += Math.cos(angle) * force;
            this.vy += Math.sin(angle) * force;
        }
    }
    draw(ctx) {
        ctx.save();
        ctx.fillStyle = this.color;
        ctx.shadowBlur = 10;
        ctx.shadowColor = this.color;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
    }
}

class Cloud {
    constructor(w, h) {
        this.w = w; this.h = h;
        this.x = Math.random() * w;
        this.y = Math.random() * (h * 0.35);
        this.size = Math.random() * 80 + 80;
        this.speed = Math.random() * 0.2 + 0.05;
    }
    update() {
        this.x += this.speed;
        if (this.x - this.size > this.w) this.x = -this.size;
    }
    draw(ctx) {
        ctx.save();
        ctx.fillStyle = 'rgba(255, 255, 255, 0.04)';
        ctx.filter = 'blur(35px)';
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.arc(this.x + this.size * 0.4, this.y - 20, this.size * 0.7, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
    }
}

class BabyCharacter {
    constructor(w, h, isLeader = false) {
        this.w = w; this.h = h;
        this.x = Math.random() * w;
        this.y = Math.random() * h;
        this.size = Math.random() * 20 + 50;
        this.isLeader = isLeader;
        this.hatColor = ['#6366f1', '#ec4899', '#f59e0b', '#10b981', '#3b82f6'][Math.floor(Math.random() * 5)];
        this.skinColor = ['#ffdbac', '#f1c27d', '#e0ac69', '#8d5524'][Math.floor(Math.random() * 4)];
        this.vx = (Math.random() - 0.5) * 2;
        this.vy = (Math.random() - 0.5) * 2;
        this.angle = 0;
    }
    update(w, h, mouse) {
        if (this.isLeader) {
            // Leader follows mouse
            const dx = mouse.x - this.x;
            const dy = mouse.y - this.y;
            this.x += dx * 0.035;
            this.y += dy * 0.035;
        } else {
            // Others wander or slowly follow the leader
            this.x += this.vx;
            this.y += this.vy;
            if (this.x < 50 || this.x > w - 50) this.vx *= -1;
            if (this.y < 50 || this.y > h - 50) this.vy *= -1;
            
            // Re-aim occasionally
            if (Math.random() > 0.99) {
                this.vx = (Math.random() - 0.5) * 2;
                this.vy = (Math.random() - 0.5) * 2;
            }
        }
        this.angle = Math.sin(Date.now() * 0.005 + this.x * 0.01) * 0.15;
    }
    draw(ctx) {
        ctx.save();
        ctx.translate(this.x, this.y);
        ctx.rotate(this.angle);
        ctx.shadowBlur = 20;
        ctx.shadowColor = this.isLeader ? 'rgba(255, 255, 255, 0.4)' : 'rgba(219, 234, 254, 0.2)';
        
        // Head
        ctx.fillStyle = this.skinColor;
        ctx.beginPath();
        ctx.arc(0, 0, this.size/2, 0, Math.PI * 2);
        ctx.fill();
        
        // Eyes
        ctx.fillStyle = '#333';
        ctx.beginPath();
        ctx.arc(-8, -4, 3, 0, Math.PI * 2);
        ctx.arc(8, -4, 3, 0, Math.PI * 2);
        ctx.fill();
        
        // Smile
        ctx.strokeStyle = '#ff5e5e';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.arc(0, 4, 8, 0, Math.PI);
        ctx.stroke();
        
        // Hat
        ctx.fillStyle = this.hatColor;
        ctx.beginPath();
        ctx.moveTo(-this.size/2, -5);
        ctx.quadraticCurveTo(0, -this.size/1.3, this.size/2, -5);
        ctx.fill();
        ctx.restore();
    }
}

class HyperModernBackground {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d', { alpha: false });
    this.blobs = [];
    this.toys = [];
    this.clouds = [];
    this.babies = [];
    this.mouse = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
    this.resize();
    this.init();
    this.bindEvents();
    this.animate();
  }

  resize() {
    this.width = window.innerWidth;
    this.height = window.innerHeight;
    this.canvas.width = this.width;
    this.canvas.height = this.height;
  }

  init() {
    this.blobs = [];
    const colors = ['#6366f1', '#06b6d4', '#8b5cf6', '#ec4899'];
    for (let i = 0; i < 4; i++) {
      this.blobs.push({
        x: Math.random() * this.width,
        y: Math.random() * this.height,
        r: Math.random() * 500 + 400,
        color: colors[i % colors.length],
        vx: (Math.random() - 0.5) * 0.2,
        vy: (Math.random() - 0.5) * 0.2
      });
    }

    this.clouds = [];
    for(let i=0; i<6; i++) this.clouds.push(new Cloud(this.width, this.height));
    
    // Multiple Babies!
    this.babies = [];
    for(let i=0; i<5; i++) {
        this.babies.push(new BabyCharacter(this.width, this.height, i === 0));
    }

    this.toys = [];
    for (let i = 0; i < 10; i++) this.toys.push(new Toy(this.width, this.height));
  }

  bindEvents() {
    window.addEventListener('resize', () => this.resize());
    document.addEventListener('mousemove', (e) => {
      this.mouse.x = e.clientX;
      this.mouse.y = e.clientY;
    });
    this.canvas.addEventListener('mousedown', (e) => {
        if (this.toys.length > 25) this.toys.shift();
        this.toys.push(new Toy(this.width, this.height, e.clientX, e.clientY));
    });
  }

  animate() {
    this.ctx.fillStyle = '#05070a';
    this.ctx.fillRect(0, 0, this.width, this.height);

    this.ctx.save();
    this.ctx.globalCompositeOperation = 'screen';
    this.ctx.filter = 'blur(130px)';
    this.blobs.forEach(b => {
        b.x += b.vx; b.y += b.vy;
        if (b.x < -100 || b.x > this.width + 100) b.vx *= -1;
        if (b.y < -100 || b.y > this.height + 100) b.vy *= -1;
        const g = this.ctx.createRadialGradient(b.x, b.y, 0, b.x, b.y, b.r);
        g.addColorStop(0, b.color + '15');
        g.addColorStop(1, 'transparent');
        this.ctx.fillStyle = g;
        this.ctx.beginPath();
        this.ctx.arc(b.x, b.y, b.r, 0, Math.PI * 2);
        this.ctx.fill();
    });
    this.ctx.restore();

    this.clouds.forEach(c => { c.update(); c.draw(this.ctx); });
    this.toys.forEach(t => { t.update(this.width, this.height, this.mouse); t.draw(this.ctx); });
    
    // Draw all babies
    this.babies.forEach(b => {
        b.update(this.width, this.height, this.mouse);
        b.draw(this.ctx);
    });

    requestAnimationFrame(() => this.animate());
  }
}

// ── i18n Translation Engine (Unchanged) ──
const dictionary = {
  'hi': {
    'home_title': 'बिलीरुबिन स्क्रीनिंग — तुरंत और सुलभ',
    'process_screening': '🔬 स्क्रीनिंग शुरू करें',
    'meet_dev': 'डेवलपर से मिलें',
    'new_session': '🩺 नया स्क्रीनिंग सत्र',
    'newborn_name': 'नवजात का नाम',
    'parent_name': 'माता-पिता / अभिभावक',
    'age': 'उम्र (घंटे)',
    'blood': 'रक्त प्रकार',
    'run_ai': '🚀 एआई विश्लेषण चलाएं',
    'tech_showcase': '🔬 प्रौद्योगिकी प्रदर्शन',
    'diagnosis_summary': '🩺 निदान सारांश',
    'key_metrics': '🩺 मुख्य मेट्रिक्स',
    'risk_level': 'जोखिम स्तर',
    'how_to_use': '📖 उपयोग कैसे करें',
    'records_title': 'मरीजों का रिकॉर्ड',
    'dashboard_title': 'अस्पताल डैशबोर्ड',
    'estimated_bilirubin': 'अनुमानित बिलीरुबिन',
    'clinical_rec': '📋 क्लिनिकल सिफारिश',
    'attention_map': '🧠 एआई अटेंशन मैप (Grad-CAM)',
    'class_probs': '📊 वर्ग संभावनाएं'
  }
};

function initTranslations() {
    const lang = document.documentElement.lang;
    if (lang === 'en' || !dictionary[lang]) return;
    const dict = dictionary[lang];
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (dict[key]) {
            if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
                el.placeholder = dict[key];
            } else {
                el.textContent = dict[key];
            }
        }
    });
}

document.addEventListener('DOMContentLoaded', () => {
    new HyperModernBackground('particles-canvas');
    initTranslations();
    const toggle = document.getElementById('nav-toggle');
    const links = document.getElementById('nav-links');
    if (toggle && links) toggle.onclick = () => links.classList.toggle('open');
});
