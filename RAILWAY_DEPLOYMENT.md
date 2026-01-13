# 🚀 Railway Deployment Guide

## Complete Step-by-Step Guide to Deploy Your RAG App

You've already connected your GitHub repo to Railway! Now follow these steps:

---

## 📋 Step 1: Configure Railway Project

### 1.1 Go to Your Railway Dashboard
- Visit: https://railway.app/dashboard
- Find your project: **RAG-FYP-Handbook-Assistant**

### 1.2 Set Environment Variables

Click on your project → **Variables** tab → Add the following:

```bash
# Required
GROQ_API_KEY=your_actual_groq_api_key_here

# Optional (for production)
ENVIRONMENT=production
GROQ_MODEL=llama-3.1-8b-instant
LLM_MAX_TOKENS=1500
LLM_TEMPERATURE=0.3
```

**IMPORTANT:** Copy your Groq API key from your local `.env` file!

---

## 📋 Step 2: Upload Required Files

Since these files are in `.gitignore`, you need to handle them specially:

### Option A: Generate on Railway (Recommended)

1. **Upload the PDF to Railway:**
   - In Railway dashboard → **Settings** → **Volumes**
   - Create a new volume (optional, or use the file system)
   - Or commit the PDF temporarily:
     ```bash
     # On your local machine
     git add -f FYP-Handbook-2023.pdf
     git commit -m "Add PDF for deployment"
     git push origin main
     ```
   - **Note:** You can remove it later with:
     ```bash
     git rm --cached FYP-Handbook-2023.pdf
     git commit -m "Remove PDF from repo"
     git push origin main
     ```

2. **Railway will automatically run ingestion:**
   - The `railway_init.sh` script will check for index files
   - If missing, it will automatically run `python ingest.py`
   - This happens on first deployment

### Option B: Upload Pre-generated Files

If you want faster deployments:

```bash
# Temporarily add generated files
git add -f faiss_index.bin chunks_metadata.pkl config.json FYP-Handbook-2023.pdf
git commit -m "Add generated files for deployment"
git push origin main

# Railway will detect and use them
# You can remove them later from git
```

---

## 📋 Step 3: Configure Build Settings

In Railway dashboard:

1. **Build Command:** (Leave default or empty - Railway auto-detects)
2. **Start Command:** Already configured in `Procfile`
   ```
   streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
   ```

---

## 📋 Step 4: Deploy!

### 4.1 Trigger Deployment

Railway should automatically deploy when you push to `main`. If not:

1. Go to your Railway project
2. Click **Deploy** button
3. Select **main** branch
4. Click **Deploy**

### 4.2 Monitor Deployment

Watch the build logs in Railway:
- Building dependencies (~3-5 minutes first time)
- Running initialization (if PDF present)
- Starting Streamlit

### 4.3 Get Your Public URL

Once deployed:
1. Go to **Settings** → **Networking**
2. Click **Generate Domain**
3. Railway will give you a URL like:
   ```
   https://your-app-name.up.railway.app
   ```

---

## ✅ Step 5: Test Your Deployment

Visit your Railway URL and test:

1. **App loads** ✓
2. **System info shows in sidebar** ✓
3. **Ask a question** ✓
4. **Get LLM-powered answer with citations** ✓

---

## 🔧 Troubleshooting

### Error: "PDF file not found"

**Solution:** Add PDF to your repo temporarily:
```bash
git add -f FYP-Handbook-2023.pdf
git commit -m "Add PDF for deployment"
git push origin main
```

### Error: "Groq API key not found"

**Solution:** Add environment variable in Railway:
1. Railway Dashboard → Your Project → Variables
2. Add: `GROQ_API_KEY=your_key`
3. Redeploy

### Error: "Module not found"

**Solution:** Railway might be caching. Try:
1. Railway → Settings → Reset Build Cache
2. Redeploy

### App is slow/timing out

**Solution:** 
1. Check Railway logs for errors
2. Upgrade Railway plan if needed (free tier has limits)
3. Consider using pre-generated index files

### Build takes too long

**Solution:** Upload pre-generated files instead of generating on Railway:
```bash
git add -f faiss_index.bin chunks_metadata.pkl config.json
git commit -m "Add pre-generated index"
git push origin main
```

---

## 📊 Environment Configuration

Your app now supports dev/prod configs via `config_manager.py`:

### Development (Local)
```bash
ENVIRONMENT=development
DEBUG=True
```

### Production (Railway)
```bash
ENVIRONMENT=production
DEBUG=False
```

Railway automatically sets `ENVIRONMENT=production`.

---

## 💰 Railway Free Tier Limits

- **500 hours/month** (enough for testing)
- **100 GB bandwidth**
- **1 GB RAM**
- **1 GB disk**

For production, consider upgrading to **Pro** ($5/month).

---

## 🎯 Quick Deployment Checklist

- [ ] GitHub repo connected to Railway
- [ ] Environment variables set (GROQ_API_KEY)
- [ ] PDF uploaded or committed
- [ ] Deployment triggered
- [ ] Public URL generated
- [ ] App tested and working

---

## 📝 Post-Deployment

### Update Your Repository

After successful deployment, you might want to:

1. **Remove PDF from git** (if you committed it):
   ```bash
   git rm --cached FYP-Handbook-2023.pdf
   git commit -m "Remove PDF from repo (deployed to Railway)"
   git push origin main
   ```

2. **Update README** with your live URL:
   ```markdown
   ## 🌐 Live Demo
   Try it: https://your-app.up.railway.app
   ```

3. **Monitor usage** in Railway dashboard

---

## 🔒 Security Notes

✅ **Safe:**
- `.env` is in `.gitignore` (not pushed)
- Environment variables in Railway (secure)
- API keys never in code

❌ **Never commit:**
- `.env` file with real keys
- API keys in code
- Sensitive credentials

---

## 🎉 You're Done!

Your RAG app is now live on Railway with:
- ✅ Public URL
- ✅ Groq LLM integration
- ✅ Automatic deployments on push
- ✅ Dev/Prod configuration
- ✅ Secure environment variables

Share your URL and get feedback! 🚀

---

## 📞 Need Help?

- **Railway Docs:** https://docs.railway.app
- **Railway Discord:** https://discord.gg/railway
- **Your Logs:** Railway Dashboard → Deployments → View Logs

---

**Estimated Setup Time:** 10-15 minutes (first deployment might take longer due to building)
