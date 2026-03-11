# 🎯 Dashboard Guide

## Quick Start - Launch Dashboard

### Step 1: Install Streamlit
```bash
pip install streamlit plotly
```

### Step 2: Launch Dashboard
```bash
# Option 1: Using launcher script
python run_dashboard.py

# Option 2: Direct streamlit command
streamlit run app.py

# Option 3: With custom port
streamlit run app.py --server.port 8502
```

### Step 3: Access Dashboard
The dashboard will automatically open in your browser at:
```
http://localhost:8501
```

---

## 🎨 Dashboard Features

### 1. Home Page 🏠
- **Project Overview**: System status and metrics
- **Quick Start Guide**: Setup instructions
- **Architecture Diagram**: System components
- **Feature List**: All capabilities

### 2. Dataset Page 📊
- **Dataset Statistics**: Train/val/test split info
- **Category Distribution**: Visual charts
- **Sample Data**: View actual tickets
- **Text Analysis**: Length distribution

### 3. Prediction Page 🤖
- **Real-time Classification**: Enter tickets and get predictions
- **Model Selection**: Choose between different models
- **Top-K Predictions**: Get ranked results
- **Example Tickets**: Quick test cases
- **Confidence Scores**: Visual progress bars

### 4. Analytics Page 📈
- **Model Comparison**: Performance metrics
- **Accuracy Charts**: Visual comparisons
- **Training Curves**: Loss over time
- **Performance Metrics**: F1, Precision, Recall

### 5. Pipeline Page ⚙️
- **Step-by-Step Execution**: Run each pipeline step
- **Command Reference**: Copy-paste commands
- **Status Tracking**: See what's completed
- **Quick Actions**: One-click operations

---

## 🚀 Complete Workflow

### First Time Setup

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Download Dataset**
```bash
python src/dataset_downloader.py
```

3. **Preprocess Data**
```bash
python src/data_preprocessing.py
```

4. **Launch Dashboard**
```bash
streamlit run app.py
```

### Using the Dashboard

1. **Navigate** using the sidebar menu
2. **View Dataset** statistics and distributions
3. **Make Predictions** on the Predict page
4. **Analyze Performance** on Analytics page
5. **Run Pipeline** steps from Pipeline page

---

## 📊 Dashboard Pages Explained

### Home Page
- Shows system status
- Displays key metrics
- Provides quick start guide
- Shows architecture

### Dataset Page
- **Metrics**: Total samples, split sizes
- **Charts**: Category distribution (bar & pie)
- **Table**: Sample tickets
- **Analysis**: Text length statistics

### Prediction Page
- **Input**: Text area for ticket description
- **Settings**: Model type and top-K selection
- **Results**: Ranked predictions with confidence
- **Examples**: Pre-filled test cases

### Analytics Page
- **Comparison**: Zero-shot vs Few-shot vs Fine-tuned
- **Metrics**: Accuracy, F1, Precision, Recall
- **Visualizations**: Bar charts, line graphs
- **Insights**: Best model recommendations

### Pipeline Page
- **6 Steps**: Complete ML pipeline
- **Commands**: Ready-to-run code
- **Descriptions**: What each step does
- **Actions**: Quick execution buttons

---

## 🎯 Example Usage

### Classify a Ticket

1. Go to **🤖 Predict** page
2. Enter ticket text:
   ```
   My laptop overheats and shuts down automatically
   ```
3. Select model type
4. Click **🎯 Classify Ticket**
5. View top 3 predictions with confidence scores

### View Dataset Stats

1. Go to **📊 Dataset** page
2. See total samples and split sizes
3. View category distribution charts
4. Browse sample tickets
5. Analyze text length distribution

### Compare Models

1. Go to **📈 Analytics** page
2. View accuracy comparison chart
3. Check performance metrics table
4. See training loss curve
5. Review model insights

---

## 🛠️ Troubleshooting

### Dashboard Won't Start

**Problem**: `streamlit: command not found`

**Solution**:
```bash
pip install streamlit
```

### Port Already in Use

**Problem**: Port 8501 is busy

**Solution**:
```bash
streamlit run app.py --server.port 8502
```

### Data Not Found

**Problem**: "Dataset not found" warning

**Solution**:
```bash
python src/dataset_downloader.py
python src/data_preprocessing.py
```

### Slow Performance

**Problem**: Dashboard is slow

**Solution**:
- Use rule-based model for faster predictions
- Reduce dataset size for testing
- Close other applications

---

## 🎨 Customization

### Change Theme

Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### Modify Port

```bash
streamlit run app.py --server.port YOUR_PORT
```

### Add Custom Pages

Edit `app.py` and add new functions:
```python
def show_custom_page():
    st.markdown("## My Custom Page")
    # Your code here
```

---

## 📱 Mobile Access

Access dashboard from mobile devices:

1. Find your computer's IP address
2. Open browser on mobile
3. Navigate to: `http://YOUR_IP:8501`

---

## 🔒 Security Notes

### Local Development
- Dashboard runs on localhost by default
- Only accessible from your computer

### Production Deployment
- Use authentication (Streamlit Cloud)
- Enable HTTPS
- Set up firewall rules
- Use environment variables for secrets

---

## 📚 Additional Resources

### Streamlit Documentation
- [Official Docs](https://docs.streamlit.io)
- [API Reference](https://docs.streamlit.io/library/api-reference)
- [Gallery](https://streamlit.io/gallery)

### Project Documentation
- `README.md` - Project overview
- `DOCUMENTATION.md` - Technical details
- `QUICKSTART.md` - Quick start guide
- `DEPLOYMENT.md` - Production deployment

---

## 🎉 Features Checklist

- ✅ Interactive web interface
- ✅ Real-time predictions
- ✅ Visual analytics
- ✅ Dataset exploration
- ✅ Model comparison
- ✅ Pipeline execution
- ✅ Responsive design
- ✅ Professional UI
- ✅ Easy navigation
- ✅ Example tickets

---

## 💡 Tips

1. **First Time**: Run data pipeline before using dashboard
2. **Performance**: Use rule-based model for instant results
3. **Exploration**: Check Dataset page for insights
4. **Testing**: Use example tickets on Predict page
5. **Comparison**: View Analytics for model performance

---

## 🚀 Next Steps

1. Launch dashboard: `streamlit run app.py`
2. Explore all pages
3. Test predictions
4. View analytics
5. Run pipeline steps
6. Deploy to production (optional)

---

**Enjoy your ML Dashboard! 🎯**
