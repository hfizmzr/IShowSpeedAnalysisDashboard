let monthlyChart;
let sentimentChart;
let sentimentChart2;
let sentimentChart3;
let sentimentChart4;
let sentimentChart5;
let predictionChart;
let categoryChart;
let sentimentTrendChart;
let reactionChart;
let comparisonChart;
let engagementComparisonChart;
let sentimentComparisonChart;
let frequencyChart;
let rawTrendLabels = [];
let countryPredictionChart;
let tweetChart;
let tweetComparisonChart;
const countryColors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#6366f1', '#8b5cf6', '#ec4899', '#06b6d4'];


// Define API_BASE_URL in the global scope
// const API_BASE_URL = `${window.location.protocol}//${window.location.hostname}/api`;
// const API_BASE_URL = `https://ishowspeedanalysisdashboard.onrender.com/api`;
const API_BASE_URL = window.location.hostname === "localhost"
  ? "http://localhost:5000/api"
  : "https://ishowspeed-backend.onrender.com/api";


document.addEventListener('DOMContentLoaded', async function() {
    // Initialize charts
    initializeMonthlyChart();
    initializeSentimentChart();
    initializeSentimentTrendCharts1();
    initializeCategoryChart();
    initializePredictionChart();
    initializeCountryPredictionChart();
    initializeComparisonCharts();
    initializeReactionChart();
    initializeTweetCategoryChart();
    initializeTweetComparisonCategoryChart();
    
    // Load and display overview info
    const overviewData = await fetchChannelData();
    const engagementData = await fetchTotalEngagementRate();
    updateChannelOverview(overviewData, engagementData);

    // Load default Monthly Views data
    const initialMonthlyData = await fetchMonthlyViews();
    updateMonthlyChart(initialMonthlyData , 'Monthly Views', '#3b82f6');

    // Attach dropdown change listener AFTER DOM is loaded
    const chartSelector = document.querySelector('select');
    chartSelector.addEventListener('change', async function () {
        const selection = this.value;
        let data, label, color;

        switch (selection) {
            case 'Monthly Views':
                data = await fetchMonthlyViews();
                label = 'Monthly Views';
                color = '#3b82f6';
                break;
            case 'Monthly Growth Views':
                data = await fetchMonthlyGrowth();
                label = 'Monthly Growth (%)';
                color = '#10b981';
                break;
            case 'Monthly Likes':
                data = await fetchMonthlyLikes();
                label = 'Monthly Likes';
                color = '#f97316';
                break;
            case 'Monthly Growth Likes':
                data = await fetchMonthlyGrowthLikes();
                label = 'Growth in Likes (%)';
                color = '#facc15';
                break;
            default:
                console.warn('Unknown selection:', selection);
                return;
        }
        
        const titleElement = document.getElementById('monthlyChartTitle');
        if (titleElement) {
            titleElement.textContent = selection;
        }

        updateMonthlyChart(data, label, color);
    });

    // Load and display top video data
    const initialTopVideoData = await fetchTop3Videos();
    updateVideoData(initialTopVideoData);

    // Load and display sentiment analysis data
    const initialSentimentData2 = await fetchSentimentSummary();
    updateSentimentCharts2(initialSentimentData2);
    const initialSentimentData3 = await fetchTiktokSentimentSummary();
    updateSentimentCharts3(initialSentimentData3);
    const initialSentimentData4 = await fetchTwitterSentimentSummary();
    updateSentimentCharts4(initialSentimentData4);
    const initialSentimentData5 = await fetchIgSentimentSummary();
    updateSentimentCharts5(initialSentimentData5);
    sentimentArray = [initialSentimentData2, initialSentimentData3, initialSentimentData4, initialSentimentData5];
    updateSentimentCharts(sentimentArray);

    // Load and display top sentiment comments
    const initialCommentsData = await fetchTopSentimentComments();
    updateComments(initialCommentsData);
    
    // Load and display sentiment trend data
    const initialSentimentTrend = await fetchMonthlySentimentTrend();
    //updateSentimentTrendCharts(initialSentimentTrend);
    const initalSentimentTrend1 = await fetchSmoothedSentimentTrend();
    updateSentimentTrendCharts1(initalSentimentTrend1);

    // Load and display category data
    const initialCategoryData = await fetchContentEngagement();
    updateCategoryChart(initialCategoryData);

    const initialReactionData = await fetchReaction();
    updateReactionChart(initialReactionData);

    const initialTweetCategory = await fetchTwitterEngagement();
    updateTweetCategoryChart(initialTweetCategory);
    updateTweetComparisonChart(initialTweetCategory);

    // Load and display comparison charts
    const initialComparisonData1 = await fetchMonthlyViews();
    const initialComparisonData2 = await fetchKaiCenatMonthlyViews();
    const initialComparisonData3 = await fetchMrBeastMonthlyViews();
    const initialComparisonData4 = await fetchPewdiepieMonthlyViews()
    updateComparisonCharts(initialComparisonData1, initialComparisonData2, initialComparisonData3, initialComparisonData4);

    // Load and display comparison charts
    const initialEngagementComparisonData1 = await fetchIShowSpeedSummary();
    const initialEngagementComparisonData2 = await fetchKaiCenatSummary();
    const initialEngagementComparisonData3 = await fetchMrBeastSummary();
    const initialEngagementComparisonData4 = await fetchPewdiepieSummary();
    updateEngagementComparisonCharts(initialEngagementComparisonData1, initialEngagementComparisonData2, initialEngagementComparisonData3, initialEngagementComparisonData4);

    // Load and display comparison charts   
    const initialSentimentComparisonData1 = await fetchIShowSpeedVader();
    const initialSentimentComparisonData2 = await fetchKaiCenatVader();
    const initialSentimentComparisonData3 = await fetchMrBeastVader();
    const initialSentimentComparisonData4 = await fetchPewdiepieVader();
    updateSentimentComparisonCharts(initialSentimentComparisonData1, initialSentimentComparisonData2, initialSentimentComparisonData3, initialSentimentComparisonData4);

    // Load and display comparison charts
    const initialFrequencyComparisonData1 = await fetchIShowSpeedPostingFreq();
    const initialFrequencyComparisonData2 = await fetchKaiCenatPostingFreq();
    const initialFrequencyComparisonData3 = await fetchMrBeastPostingFreq();
    const initialFrequencyComparisonData4 = await fetchPewdiepiePostingFreq();
    updateFrequencyComparisonCharts(initialFrequencyComparisonData1, initialFrequencyComparisonData2, initialFrequencyComparisonData3, initialFrequencyComparisonData4);

    // Load and display forecast view data
    const initialForecastData = await fetchForecastedViews();
    const initialForecastDataArima = await fetchForecastedViewsArima();
    const initialForecastDataSarima = await fetchForecastedViewsSarima();
    const initialForecastDataProphet = await fetchForecastedViewsProphet();
    const initialForecastDataMonthly = await fetchForecastedViewsMonthly();
    updatePredictionChart(initialForecastDataProphet);
    // updatePredictionChart2(initialForecastDataMonthly);

    // Load and display country prediction data
    const initialCountryData = await fetchPredictedCountries();
    updateCountryPredictionChart(initialCountryData);

    const initialOpportunitiesData = await fetchOpportunities();
    updateOpportunities(initialOpportunitiesData);
});

// Fetch data from API
async function fetchData(endpoint, label) {
    try {
        const url = `${API_BASE_URL}/${endpoint}`;
        console.log(`Fetching ${label} from:`, url);
        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        console.log(`${label} received:`, data);
        return data;
    } catch (error) {
        console.error(`Error fetching ${label}:`, error);
    }
}

async function fetchChannelData() {
    return fetchData("channel_data", "Channel Data");
}

async function fetchContentEngagement() {
    return fetchData("content_engagement", "Content Engagement");
}

async function fetchContentSentiment() {
    return fetchData("content_sentiment", "Content Sentiment");
}

async function fetchForecastedViews() {
    return fetchData("forecasted_views", "Forecasted Views");
}

async function fetchForecastedViewsArima() {
    return fetchData("forecasted_views_arima", "Forecasted Views");
}

async function fetchForecastedViewsSarima() {
    return fetchData("forecasted_views_sarima", "Forecasted Views");
}

async function fetchForecastedViewsProphet() {
    return fetchData("forecasted_views_prophet", "Forecasted Views");
}

async function fetchMonthlyGrowthLikes() {
    return fetchData("monthly_growth_likes", "Monthly Growth Likes");
}

async function fetchMonthlyGrowth() {
    return fetchData("monthly_growth", "Monthly Growth");
}

async function fetchMonthlyLikes() {
    return fetchData("monthly_likes", "Monthly Likes");
}

async function fetchMonthlyViews() {
    return fetchData("monthly_views", "Monthly Views");
}

async function fetchPredictedCountries() {
    return fetchData("predicted_countries", "Predicted Countries");
}

async function fetchSentimentSummary() {
    return fetchData("sentiment_summary", "Sentiment Summary");
}

async function fetchSmoothedSentimentTrend() {
    return fetchData("smoothed_sentiment_trend", "Smoothed Sentiment Trend");
}

async function fetchMonthlySentimentTrend() {
    return fetchData("monthly_sentiment_trend", "Monthly Sentiment Trend");
}

async function fetchTop3Videos() {
    return fetchData("top_3_videos", "Top 3 Videos");
}

async function fetchTopSentimentComments() {
    return fetchData("top_sentiment_comments", "Top Sentiment Comments");
}

async function fetchTotalEngagementRate() {
    return fetchData("total_engagement_rate", "Total Engagement Rate");
}

async function fetchIShowSpeedSummary() {
    return fetchData("ishowspeed_summary", "IShowSpeed Summary");
}

async function fetchKaiCenatSummary() {
    return fetchData("kaicenat_summary", "Kai Cenat Summary");
}

async function fetchMrBeastSummary() {
    return fetchData("mrbeast_summary", "MrBeast Summary");
}

async function fetchPewdiepieSummary() {
    return fetchData("pewdiepie_summary", "Pewdiepie Summary");
}

async function fetchIShowSpeedEngagement() {
    return fetchData("ishowspeed_engagement", "IShowSpeed Engagement");
}

async function fetchKaiCenatEngagement() {
    return fetchData("kaicenat_engagement", "Kai Cenat Engagement");
}

async function fetchMrBeastEngagement() {
    return fetchData("mrbeast_engagement", "MrBeast Engagement");
}

async function fetchPewdiepieEngagement() {
    return fetchData("pewdiepie_engagement", "Pewdiepie Engagement");
}

async function fetchIShowSpeedPostingFreq() {
    return fetchData("ishowspeed_posting_freq", "IShowSpeed Posting Frequency");
}

async function fetchKaiCenatPostingFreq() {
    return fetchData("kaicenat_posting_freq", "Kai Cenat Posting Frequency");
}

async function fetchMrBeastPostingFreq() {
    return fetchData("mrbeast_posting_freq", "MrBeast Posting Frequency");
}

async function fetchPewdiepiePostingFreq() {
    return fetchData("pewdiepie_posting_freq", "Pewdiepiw Posting Frequency");
}

async function fetchIShowSpeedVader() {
    return fetchData("ishowspeed_vader", "IShowSpeed Vader Sentiment");
}

async function fetchKaiCenatVader() {
    return fetchData("kaicenat_vader", "Kai Cenat Vader Sentiment");
}

async function fetchMrBeastVader() {
    return fetchData("mrbeast_vader", "MrBeast Vader Sentiment");
}

async function fetchPewdiepieVader() {
    return fetchData("pewdiepie_vader", "Pewdiepie Vader Sentiment");
}

async function fetchMrBeastMonthlyViews() {
    return fetchData("mrbeast_monthly_views", "MrBeast Monthly Views");
}

async function fetchKaiCenatMonthlyViews() {
    return fetchData("kaicenat_monthly_views", "Kai Cenat Monthly Views");
}

async function fetchPewdiepieMonthlyViews() {
    return fetchData("pewdiepie_monthly_views", "Pewdiepie Monthly Views");
}

async function fetchReaction() {
    return fetchData("content_sentiment_percentage", "Reaction");
}

async function fetchOpportunities() {
    return fetchData("explore_growth_opportunities", "Explore Growth Opportunities");
}

async function fetchForecastedViewsMonthly() {
    return fetchData("combined_6mo_forecast", "Forecast");
}

async function fetchTiktokSentimentSummary() {
    return fetchData("tiktok_sentiment_results", "Tiktok Sentiment Summary");
}

async function fetchTwitterSentimentSummary() {
    return fetchData("twitter_sentiment_results", "Twitter Sentiment Summary");
}

async function fetchIgSentimentSummary() {
    return fetchData("ig_sentiment_results", "Ig Sentiment Summary");
}

async function fetchTwitterEngagement() {
    return fetchData("twitter_avg_engagement", "Twitter Engagement");
}

// Update channel overview with data
function updateChannelOverview(data, data2) {
    if (!data || !data.channel_data) {
        console.log('No channel info data available');
        return;
    }

    if (!data2 || typeof data2.total_engagement_rate === 'undefined') {
        console.log('No engagement rate data available');
        return;
    }
    
    console.log('Updating channel overview with:', data.channel_data, data2.total_engagement_rate);
    const channelInfo = data.channel_data;
    const totalEngagementRate = data2.total_engagement_rate;
    
    const subscriberElement = document.getElementById('subscribersCount');
    if (subscriberElement) {
        subscriberElement.textContent = formatNumber(channelInfo.subscriber_count || 0);
        console.log('Updated subscriber count to:', subscriberElement.textContent);
    } else {
        console.log('Subscriber element not found');
    }

    const viewsElement = document.getElementById('viewsCount');
    if (viewsElement) {
        viewsElement.textContent = formatNumber(channelInfo.total_view_count || 0);
        console.log('Updated views count to:', viewsElement.textContent);
    } else {
        console.log('Views element not found');
    }

    const descriptionElement = document.getElementById('channelDescription');
    if (descriptionElement && channelInfo.description) {
        descriptionElement.textContent = channelInfo.description;
        console.log('Updated channel description');
    } else {
        console.log('Description element not found or no description available');
    }

    const engagementElement = document.getElementById('engagementRate');
    if (engagementElement && totalEngagementRate) {
        engagementElement.textContent = `${totalEngagementRate}%`;
        console.log('Updated channel description');
    } else {
        console.log('Description element not found or no description available');
    }
}

// Update video data in the UI
function updateVideoData(data) {
    if (!data || !data.length) return;

    // Sort videos by view count
    const sortedVideos = [...data].sort((a, b) => b.view_count - a.view_count);
    const topVideos = sortedVideos.slice(0, 3);

    // Get the container for top performing videos
    const topVideosContainer = document.querySelector('.bg-gray-800 .space-y-4');
    if (!topVideosContainer) return;

    // Clear existing content
    topVideosContainer.innerHTML = '';

    // Add top videos
    topVideos.forEach((video, index) => {
        const videoElement = document.createElement('div');
        videoElement.className = 'flex items-center justify-between p-3 bg-gray-700 rounded-lg';

        videoElement.innerHTML = `
            <div class="flex items-center space-x-3">
                <a href="${video.url}" target="_blank" rel="noopener noreferrer" aria-label="Watch ${video.title}">
                    <img src="${video.thumbnail || 'https://via.placeholder.com/60'}" alt="Thumbnail of ${video.title}" class="w-12 h-12 rounded-md">
                </a>
                <div>
                    <a href="${video.url}" target="_blank" rel="noopener noreferrer" class="font-medium hover:underline">
                        ${video.title}
                    </a>
                    <p class="text-sm text-gray-400">${formatNumber(video.view_count)} views</p>
                </div>
            </div>
            <span class="bg-blue-600 text-xs px-2 py-1 rounded-full">Top ${index + 1}</span>
        `;

        topVideosContainer.appendChild(videoElement);
    });
}


// initialize Monthly Chart
function initializeMonthlyChart() {
    const ctx = document.getElementById('monthlyChart').getContext('2d');
    monthlyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Monthly Data',
                data: [],
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y; // Only show the value without label
                        }
                    }
                }
            },
            scales: {
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#9ca3af',
                        callback: function(value) {
                            return formatNumber(value);
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: { 
                        color: '#9ca3af'
                    }
                }
            }
        }
    });
}

// Update monthly chart with data
function updateMonthlyChart(data, label, color = '#3b82f6') {
    if (!data || !Array.isArray(data)) {
        console.warn("Invalid data passed to chart:", data);
        return;
    }

    const shortMonthLabels = [];
    const fullMonthLabels = [];

    const values = data.map(item => {
        const [year, month] = (item.label || '').split('-');
        if (year && month) {
            const date = new Date(`${year}-${month}-01`);
            shortMonthLabels.push(date.toLocaleString('default', { month: 'short' })); // For X-axis
            fullMonthLabels.push(date.toLocaleString('default', { month: 'long', year: 'numeric' })); // For tooltip
        } else {
            shortMonthLabels.push("N/A");
            fullMonthLabels.push("N/A");
        }

        return item.value !== undefined ? item.value :
               item.growth !== undefined ? item.growth : 0;
    });

    monthlyChart.data.labels = shortMonthLabels;
    monthlyChart.data.datasets[0].label = label;
    monthlyChart.data.datasets[0].data = values;
    monthlyChart.data.datasets[0].borderColor = color;
    monthlyChart.data.datasets[0].backgroundColor = color + '1A';

    // Customize tooltips to show full month + year
    monthlyChart.options.plugins.tooltip = {
        callbacks: {
            title: function(tooltipItems) {
                return fullMonthLabels[tooltipItems[0].dataIndex];
            }
        }
    };

    monthlyChart.update();

    console.log("Chart updated with short labels:", shortMonthLabels);
    console.log("Chart tooltip full labels:", fullMonthLabels);
    console.log("Chart updated with values:", values);
}

function initializeSentimentChart() {
    const chartConfigs = [
        { id: 'sentimentChart', varName: 'sentimentChart', title: 'Overall Sentiment' },
        { id: 'sentimentChart2', varName: 'sentimentChart2', title: 'Youtube Comment' },
        { id: 'sentimentChart3', varName: 'sentimentChart3', title: 'Tiktok' },
        { id: 'sentimentChart4', varName: 'sentimentChart4', title: 'Twitter' },
        { id: 'sentimentChart5', varName: 'sentimentChart5', title: 'Instagram' },
    ];

    chartConfigs.forEach((config) => {
        const ctx = document.getElementById(config.id)?.getContext('2d');
        if (!ctx) {
            console.warn(`Canvas element with id '${config.id}' not found`);
            return;
        }

        const chart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Neutral', 'Negative'],
                datasets: [{
                    data: [0, 0, 0], // Initialize with zeros
                    backgroundColor: ['#10b981', '#ffc107', '#ef4444'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: config.title,
                        color: '#ffffff',
                        font: {
                            size: 16,
                            weight: 'bold'
                        },
                        padding: {
                            top: 10,
                            bottom: 20
                        }
                    },
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#9ca3af',
                            padding: 10,
                            usePointStyle: true,
                            pointStyle: 'circle'
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                const label = context.label || '';
                                const value = context.parsed || 0;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
                                return `${label}: ${value} (${percentage}%)`;
                            }
                        }
                    }
                },
                cutout: '70%'
            }
        });

        // Assign to global variable based on varName
        if (config.varName === 'sentimentChart') {
            sentimentChart = chart;
        } else if (config.varName === 'sentimentChart2') {
            sentimentChart2 = chart;
        } else if (config.varName === 'sentimentChart3') {
            sentimentChart3 = chart;
        } else if (config.varName === 'sentimentChart4') {
            sentimentChart4 = chart;
        } else if (config.varName === 'sentimentChart5') {
            sentimentChart5 = chart;
        }

        console.log(`Initialized ${config.varName} successfully`);
    });
}

function updateSentimentCharts(dataArray) {
    if (!sentimentChart) {
        console.warn("sentimentChart not initialized.");
        return;
    }

    if (!Array.isArray(dataArray) || dataArray.length === 0) {
        console.warn("No sentiment data provided.");
        sentimentChart.data.datasets[0].data = [0, 0, 0];
        sentimentChart.update();
        return;
    }

    let totalPositive = 0;
    let totalNeutral = 0;
    let totalNegative = 0;

    dataArray.forEach((data, index) => {
        if (!data || typeof data !== 'object') {
            console.warn(`Invalid data at index ${index}:`, data);
            return;
        }

        // Check if it's nested (e.g. data.vader)
        if ('vader' in data) {
            const { positive = 0, neutral = 0, negative = 0 } = data.vader;
            totalPositive += typeof positive === 'number' ? positive : 0;
            totalNeutral += typeof neutral === 'number' ? neutral : 0;
            totalNegative += typeof negative === 'number' ? negative : 0;
        } else {
            // Assume it's flat with capitalized keys
            const Positive = data.Positive ?? 0;
            const Neutral = data.Neutral ?? 0;
            const Negative = data.Negative ?? 0;

            totalPositive += typeof Positive === 'number' ? Positive : 0;
            totalNeutral += typeof Neutral === 'number' ? Neutral : 0;
            totalNegative += typeof Negative === 'number' ? Negative : 0;
        }
    });

    sentimentChart.data.datasets[0].data = [totalPositive, totalNeutral, totalNegative];
    sentimentChart.update();

    console.log("Combined sentiment chart updated with:", {
        positive: totalPositive,
        neutral: totalNeutral,
        negative: totalNegative
    });
}


function updateSentimentCharts2(data) {
    if (!sentimentChart2) {
        console.warn("sentimentChart2 not initialized.");
        return;
    }

    if (!data || !data.vader) {
        console.warn("No vader sentiment data found for chart 2:", data);
        // Set chart to show no data state
        sentimentChart2.data.datasets[0].data = [0, 0, 0];
        sentimentChart2.update();
        return;
    }

    const { positive = 0, neutral = 0, negative = 0 } = data.vader;
    
    // Validate that values are numbers
    const validPositive = typeof positive === 'number' && !isNaN(positive) ? positive : 0;
    const validNeutral = typeof neutral === 'number' && !isNaN(neutral) ? neutral : 0;
    const validNegative = typeof negative === 'number' && !isNaN(negative) ? negative : 0;

    sentimentChart2.data.datasets[0].data = [validPositive, validNeutral, validNegative];
    sentimentChart2.update();
    console.log("Sentiment chart 2 updated with:", { positive: validPositive, neutral: validNeutral, negative: validNegative });
}

function updateSentimentCharts3(data) {
    if (!sentimentChart3) {
        console.warn("sentimentChart3 not initialized.");
        return;
    }

    if (!data) {
        console.warn("No sentiment data found for chart 3:", data);
        // Set chart to show no data state
        sentimentChart3.data.datasets[0].data = [0, 0, 0];
        sentimentChart3.update();
        return;
    }

    // Handle different possible data structures
    let positive = 0, neutral = 0, negative = 0;
    
    if (data.Positive !== undefined) {
        positive = data.Positive;
        neutral = data.Neutral || 0;
        negative = data.Negative || 0;
    } else if (data.positive !== undefined) {
        positive = data.positive;
        neutral = data.neutral || 0;
        negative = data.negative || 0;
    }
    
    // Validate that values are numbers
    const validPositive = typeof positive === 'number' && !isNaN(positive) ? positive : 0;
    const validNeutral = typeof neutral === 'number' && !isNaN(neutral) ? neutral : 0;
    const validNegative = typeof negative === 'number' && !isNaN(negative) ? negative : 0;

    sentimentChart3.data.datasets[0].data = [validPositive, validNeutral, validNegative];
    sentimentChart3.update();
    console.log("Sentiment chart 3 updated with:", { positive: validPositive, neutral: validNeutral, negative: validNegative });
}

function updateSentimentCharts4(data) {
    if (!sentimentChart4) {
        console.warn("sentimentChart4 not initialized.");
        return;
    }

    if (!data) {
        console.warn("No sentiment data found for chart 4:", data);
        // Set chart to show no data state
        sentimentChart4.data.datasets[0].data = [0, 0, 0];
        sentimentChart4.update();
        return;
    }

    // Handle different possible data structures
    let positive = 0, neutral = 0, negative = 0;
    
    if (data.Positive !== undefined) {
        positive = data.Positive;
        neutral = data.Neutral || 0;
        negative = data.Negative || 0;
    } else if (data.positive !== undefined) {
        positive = data.positive;
        neutral = data.neutral || 0;
        negative = data.negative || 0;
    }
    
    // Validate that values are numbers
    const validPositive = typeof positive === 'number' && !isNaN(positive) ? positive : 0;
    const validNeutral = typeof neutral === 'number' && !isNaN(neutral) ? neutral : 0;
    const validNegative = typeof negative === 'number' && !isNaN(negative) ? negative : 0;

    sentimentChart4.data.datasets[0].data = [validPositive, validNeutral, validNegative];
    sentimentChart4.update();
    console.log("Sentiment chart 4 updated with:", { positive: validPositive, neutral: validNeutral, negative: validNegative });
}

function updateSentimentCharts5(data) {
    if (!sentimentChart5) {
        console.warn("sentimentChart5 not initialized.");
        return;
    }

    if (!data) {
        console.warn("No sentiment data found for chart 5:", data);
        // Set chart to show no data state
        sentimentChart5.data.datasets[0].data = [0, 0, 0];
        sentimentChart5.update();
        return;
    }

    // Handle different possible data structures
    let positive = 0, neutral = 0, negative = 0;
    
    if (data.Positive !== undefined) {
        positive = data.Positive;
        neutral = data.Neutral || 0;
        negative = data.Negative || 0;
    } else if (data.positive !== undefined) {
        positive = data.positive;
        neutral = data.neutral || 0;
        negative = data.negative || 0;
    }
    
    // Validate that values are numbers
    const validPositive = typeof positive === 'number' && !isNaN(positive) ? positive : 0;
    const validNeutral = typeof neutral === 'number' && !isNaN(neutral) ? neutral : 0;
    const validNegative = typeof negative === 'number' && !isNaN(negative) ? negative : 0;

    sentimentChart5.data.datasets[0].data = [validPositive, validNeutral, validNegative];
    sentimentChart5.update();
    console.log("Sentiment chart 5 updated with:", { positive: validPositive, neutral: validNeutral, negative: validNegative });
}

// update Comments
function updateComments(data) {
    const positiveSection = document.getElementById('positive-comments')?.parentElement;
    const negativeSection = document.getElementById('negative-comments')?.parentElement;
  
    function createCommentBlock(commentObj) {
        const container = document.createElement('div');
        container.className = 'bg-gray-700 p-3 rounded-lg';
      
        const maxLength = 20;
        const fullText = commentObj.comment;
        const isLong = fullText.length > maxLength;
        const previewText = isLong ? fullText.substring(0, maxLength) + "..." : fullText;
      
        const commentText = document.createElement('p');
        commentText.className = 'text-sm';
        commentText.textContent = previewText;
      
        const metaText = document.createElement('p');
        metaText.className = 'text-xs text-gray-400 mt-1';
        metaText.textContent = `- Video: ${commentObj.video_title}`;
      
        container.appendChild(commentText);
        container.appendChild(metaText);
      
        if (isLong) {
          const toggleButton = document.createElement('button');
          toggleButton.className = 'text-blue-400 text-xs mt-2';
          toggleButton.textContent = 'View More';
      
          let expanded = false;
          toggleButton.addEventListener('click', () => {
            expanded = !expanded;
            commentText.textContent = expanded ? fullText : previewText;
            toggleButton.textContent = expanded ? 'View Less' : 'View More';
          });
      
          container.appendChild(toggleButton);
        }
      
        return container;
      }      
  
    if (positiveSection) {
      const positiveCommentsContainer = positiveSection.querySelector('.space-y-3');
      if (positiveCommentsContainer) {
        positiveCommentsContainer.innerHTML = '';
        data.top_vader_positive.forEach(cmt => {
          positiveCommentsContainer.appendChild(createCommentBlock(cmt));
        });
      }
    }
  
    if (negativeSection) {
      const negativeCommentsContainer = negativeSection.querySelector('.space-y-3');
      if (negativeCommentsContainer) {
        negativeCommentsContainer.innerHTML = '';
        data.top_vader_negative.forEach(cmt => {
          negativeCommentsContainer.appendChild(createCommentBlock(cmt));
        });
      }
    }
}    

// Initialize sentiment trend chart
function initializeSentimentTrendChart() {
    const sentimentTrendCtx = document.getElementById('sentimentTrendChart')?.getContext('2d');
    if (sentimentTrendCtx) {
        sentimentTrendChart = new Chart(sentimentTrendCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Positive',
                        data: [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.4,
                        fill: false
                    },
                    {
                        label: 'Neutral',
                        data: [],
                        borderColor: '#3b82f6', 
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.4,
                        fill: false
                    },
                    {
                        label: 'Negative',
                        data: [],
                        color: '#9ca3af',
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        tension: 0.4,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: { 
                            color: '#9ca3af',
                            callback: function(value) {
                                return formatNumber(value);
                            }
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#9ca3af',
                            callback: function(value, index, ticks) {
                                // The label is in 'MMM YYYY', extract and return only the short month
                                const label = this.getLabelForValue(value);
                                // label is e.g., "Jan 2024", so just split and return month part
                                return label.split(' ')[0]; // returns "Jan"
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#9ca3af',
                            usePointStyle: true,
                            pointStyle: 'circle'
                        }
                    },
                    tooltip: {
                        callbacks: {
                            title: function (tooltipItems) {
                                const index = tooltipItems[0].dataIndex;
                                // Return full month and year from rawTrendLabels
                                return new Date(`${rawTrendLabels[index]}-01`).toLocaleString('default', {
                                    month: 'long',
                                    year: 'numeric'
                                });
                            }
                        }
                    }
                }
            }
        });
    }
}

// Update sentiment trend chart
function updateSentimentTrendCharts(data) {
    if (data && typeof data === 'object' && sentimentTrendChart) {
        const dates = [];
        const positives = [];
        const neutrals = [];
        const negatives = [];

        for (const [date, sentiment] of Object.entries(data)) {
            if (sentiment.positive !== undefined && sentiment.neutral !== undefined && sentiment.negative !== undefined) {
                dates.push(date);
                positives.push(sentiment.positive);
                neutrals.push(sentiment.neutral);
                negatives.push(sentiment.negative);
            }
        }

        rawTrendLabels = dates;

        sentimentTrendChart.data.labels = dates.map(date => {
            const [year, month] = date.split("-");
            return new Date(`${year}-${month}-01`).toLocaleString('default', { month: 'short', year: 'numeric' });
        });

        sentimentTrendChart.data.datasets[0].data = positives;
        sentimentTrendChart.data.datasets[1].data = neutrals;
        sentimentTrendChart.data.datasets[2].data = negatives;

        sentimentTrendChart.update();
        console.log("Sentiment trend chart updated with full sentiment data:", { dates, positives, neutrals, negatives });
    }
}

function initializeSentimentTrendCharts1() {
    const ctx = document.getElementById('sentimentTrendChart').getContext('2d');
    
    sentimentTrendChart = new Chart(ctx, {
        type: 'line', // We'll use line chart with `fill: true` for stacked areas
        data: {
            labels: [], // dates or months go here
            datasets: [
                {
                    label: 'Negative',
                    data: [],
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.6)',
                    fill: true,
                    stack: 'sentiment',
                    tension: 0.4,
                },
                {
                    label: 'Neutral',
                    data: [],
                    borderColor: '#ffc107',
                    backgroundColor: 'rgba(255, 193, 7, 0.6)',
                    fill: true,
                    stack: 'sentiment',
                    tension: 0.4,
                },
                {
                    label: 'Positive',
                    data: [],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.6)',
                    fill: true,
                    stack: 'sentiment',
                    tension: 0.4,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'month',
                        tooltipFormat: 'MMM yyyy',
                        displayFormats: {
                            month: 'MMM'
                        }
                    },
                    ticks: {
                        color: '#9ca3af',
                    },
                    grid: {
                        display: false
                    }
                },
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: value => value + '%',
                        color: '#9ca3af',
                    },
                    stacked: true,
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#9ca3af',
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y + '%'
                    }
                }
            },
            interaction: {
                mode: 'index',
                intersect: false
            }
        }
    });
}

function updateSentimentTrendCharts1(rawData) {
    if (!sentimentTrendChart) {
        console.error("Chart not initialized");
        return;
    }

    const labels = Object.keys(rawData).sort(); // Sort by date ascending

    const negativeData = [];
    const neutralData = [];
    const positiveData = [];

    labels.forEach(date => {
        const sentiment = rawData[date];
        negativeData.push(sentiment.negative);
        neutralData.push(sentiment.neutral);
        positiveData.push(sentiment.positive);
    });

    sentimentTrendChart.data.labels = labels;
    sentimentTrendChart.data.datasets[0].data = negativeData;
    sentimentTrendChart.data.datasets[1].data = neutralData;
    sentimentTrendChart.data.datasets[2].data = positiveData;

    sentimentTrendChart.update();
}

// Initialize category chart
function initializeCategoryChart() {
    const categoryCtx = document.getElementById('categoryChart').getContext('2d');
    categoryChart = new Chart(categoryCtx, {
        type: 'bar',
        data: {
            labels: ['Gaming', 'Memes', 'Challenges/IRL', 'Viral moments/Clickbait', 'Other'],
            datasets: [{
                label: 'Views',
                data: [0, 0, 0, 0, 0], // Will be updated with real data
                backgroundColor: '#3b82f6'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#9ca3af',
                        callback: function(value) {
                            return formatNumber(value);
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#9ca3af'
                    }
                }
            }
        }
    });
}

// Update category chart with data
function updateCategoryChart(data) {
    if (!Array.isArray(data) || !categoryChart) return;

    // Default label mapping
    const labelMap = {
        "Gaming": "Gaming",
        "Challenges / IRL": "Challenges/IRL",
        "Viral moments / Clickbait": "Viral moments/Clickbait",
        "Memes": "Memes",
        "Other": "Other",
    };

    // Initialize totals for each label
    const labels = ['Gaming', 'Memes', 'Challenges/IRL', 'Viral moments/Clickbait', 'Other'];
    const viewsByCategory = {
        "Gaming": 0,
        "Memes": 0,
        "Challenges/IRL": 0,
        "Viral moments/Clickbait": 0,
        "Other": 0
    };

    // Sum average views for each mapped category
    data.forEach(item => {
        const mappedLabel = labelMap[item.content_type];
        if (mappedLabel) {
            viewsByCategory[mappedLabel] += item.avg_views;
        }
    });

    // Update chart data
    categoryChart.data.labels = labels;
    categoryChart.data.datasets[0].data = labels.map(label => viewsByCategory[label]);
    categoryChart.update();

    console.log("Category chart updated:", viewsByCategory);
}

function initializeTweetCategoryChart() {
    const tweetCategoryCtx = document.getElementById('tweetChart').getContext('2d');
    tweetChart = new Chart(tweetCategoryCtx, {
        type: 'bar',
        data: {
            labels: ['Gaming', 'Memes', 'Challenges/IRL', 'Viral moments/Clickbait', 'Music', 'Other'],
            datasets: [{
                label: 'Views',
                data: [0, 0, 0, 0, 0], // Will be updated with real data
                backgroundColor: '#3b82f6'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#9ca3af',
                        callback: function(value) {
                            return formatNumber(value);
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#9ca3af'
                    }
                }
            }
        }
    });
}

function updateTweetCategoryChart(data) {
    const categoryMap = {
        'Gaming': 0,
        'Memes': 1,
        'Challenges / IRL': 2,
        'Challenges/IRL': 2, // handle both formats
        'Viral moments/Clickbait': 3,
        'Music': 4,
        'Other': 5 
    };

    // Initialize totals
    const totals = [0, 0, 0, 0, 0, 0];

    // Aggregate views by category
    const speedData = data.filter(item => item.creator === 'IShowSpeed');

    speedData.forEach(item => {
        const categoryKey = item.content_type.trim();
        const index = categoryMap[categoryKey];
        if (index !== undefined) {
            totals[index] += item.views_count;
        }
    });

    // Update the chart
    tweetChart.data.datasets[0].data = totals;
    tweetChart.update();
}

function initializeTweetComparisonCategoryChart() {
    const tweetComparisonCategoryCtx = document.getElementById('tweetComparisonChart').getContext('2d');
    tweetComparisonChart = new Chart(tweetComparisonCategoryCtx, {
        type: 'bar',
        data: {
            labels: ['Gaming', 'Memes', 'Challenges/IRL', 'Viral moments/Clickbait', 'Other'],
            datasets: [{
                label: 'Views',
                data: [0, 0, 0, 0, 0], // Will be updated with real data
                backgroundColor: '#3b82f6'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#9ca3af',
                        callback: function(value) {
                            return formatNumber(value);
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#9ca3af'
                    }
                }
            }
        }
    });
}

function updateTweetComparisonChart(data) {
    const categories = ['Gaming', 'Memes', 'Challenges/IRL', 'Viral moments/Clickbait', 'Music', 'Other'];
    const categoryMap = {
        'Gaming': 0,
        'Memes': 1,
        'Challenges/IRL': 2,
        'Challenges / IRL': 2,
        'Viral moments/Clickbait': 3,
        'Music': 4,
        'Other': 5
    };

    const creators = ['IShowSpeed', 'MrBeast'];
    const creatorData = {
        'IShowSpeed': [0, 0, 0, 0, 0, 0],
        'MrBeast': [0, 0, 0, 0, 0, 0]
    };

    // Aggregate views
    data.forEach(item => {
        const creator = item.creator;
        const type = item.content_type;
        const index = categoryMap[type];
        if (creatorData[creator] && index !== undefined) {
            creatorData[creator][index] += item.views_count;
        }
    });

    // Update chart datasets
    tweetComparisonChart.data.labels = categories;
    tweetComparisonChart.data.datasets = [
        {
            label: 'IShowSpeed',
            data: creatorData['IShowSpeed'],
            backgroundColor: '#3b82f6'
        },
        {
            label: 'MrBeast',
            data: creatorData['MrBeast'],
            backgroundColor: '#f59e0b'
        }
    ];

    tweetComparisonChart.update();
}


// Initialize prediction chart
function initializePredictionChart() {
    const predictionCtx = document.getElementById('predictionChart').getContext('2d');
    predictionChart = new Chart(predictionCtx, {
        type: 'line',
        data: {
            labels: [], // dates go here
            datasets: [
                {
                    label: 'Actual Views',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4,
                    borderWidth: 2,
                    fill: false,
                    pointRadius: 0,
                    pointHitRadius: 10,
                    pointHoverRadius: 5
                },
                {
                    label: 'Forecasted Views',
                    data: [],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4,
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0,
                    pointHitRadius: 10,
                    pointHoverRadius: 5
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#9ca3af',
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                tooltip: {
                    mode: 'nearest',
                    intersect: false,
                }
            }, scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#9ca3af',
                        callback: function(value) {
                            return formatNumber(value);
                        }
                    }
                },
                x: {
                    type: 'time',  // time scale for dates
                    time: {
                        parser: 'yyyy-MM-dd',
                        unit: 'month',               // Use month as the unit
                        displayFormats: {
                            month: 'MMM'             // Display format as abbreviated month (Jan, Feb, etc.)
                        },
                        tooltipFormat: 'MMM yyyy'    // Tooltip will show full month + year
                    },
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#9ca3af'
                    }
                }
            }
        }
    });
}

// Update prediction chart with data
function updatePredictionChart(data) {
    if (!data || !predictionChart) {
        console.error("Invalid data or chart not initialized");
        return;
    }

    // Prepare actual and forecast data points with x (date) and y (views)
    const actualData = Array.isArray(data.actual) ? data.actual.map(item => ({
        x: item.date,
        y: item.views
    })) : [];

    const forecastData = Array.isArray(data.forecast) ? data.forecast.map(item => ({
        x: item.date,
        y: item.views
    })) : [];

    if (actualData.length === 0) {
        console.warn("No actual data available");
        return;
    }

    // Merge all dates (actual + forecast) for x-axis labels
    const allDates = actualData.concat(forecastData).map(d => d.x);

    // Update chart labels to full date range
    predictionChart.data.labels = allDates;

    // Update datasets with new data
    predictionChart.data.datasets[0].data = actualData;
    predictionChart.data.datasets[1].data = forecastData;

    predictionChart.update();

    console.log("Prediction chart updated:", { actualData, forecastData });
}

function updatePredictionChart2(data) {
    if (!Array.isArray(data) || !predictionChart) {
        console.error("Invalid data or chart not initialized");
        return;
    }

    // Filter and map historical (actual) data
    const actualData = data
        .filter(item => item.type === 'historical')
        .map(item => ({
            x: item.date,
            y: item.predicted_views
        }));

    // Filter and map forecast data
    const forecastData = data
        .filter(item => item.type === 'forecast')
        .map(item => ({
            x: item.date,
            y: item.predicted_views
        }));

    if (actualData.length === 0) {
        console.warn("No historical data available");
        return;
    }

    // Merge all dates (actual + forecast) for x-axis labels
    const allDates = actualData.concat(forecastData).map(d => d.x);

    // Update chart labels
    predictionChart.data.labels = allDates;

    // Update datasets
    predictionChart.data.datasets[0].data = actualData;   // Assuming 0 is historical
    predictionChart.data.datasets[1].data = forecastData; // Assuming 1 is forecast

    predictionChart.update();

    console.log("Prediction chart updated:", { actualData, forecastData });
}


//  Initialize countries prediction chart
function initializeCountryPredictionChart() {
    const ctx = document.getElementById('countryPredictionChart').getContext('2d');
    countryPredictionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: [],
            datasets: [{
                data: [],
                backgroundColor: [],
                borderColor: '#1f2937',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            cutout: '60%',
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.parsed}`;
                        }
                    }
                }
            }
        }
    });
}

// Update countries prediction chart with data
function updateCountryPredictionChart(data) {
    if (!countryPredictionChart || !Array.isArray(data)) {
        console.error("Chart not initialized or invalid data.");
        return;
    }

    const labels = data.map(item => item.country);
    const totalScore = data.reduce((sum, item) => sum + item.score, 0);

    // Calculate percentages
    const percentages = data.map(item => (item.score / totalScore) * 100);
    const bgColors = labels.map((_, i) => countryColors[i % countryColors.length]);

    // Update chart data
    countryPredictionChart.data.labels = labels;
    countryPredictionChart.data.datasets[0].data = percentages.map(p => p.toFixed(2));
    countryPredictionChart.data.datasets[0].backgroundColor = bgColors;
    countryPredictionChart.update();

    // Update custom legend
    const legendContainer = document.getElementById('countryLegend');
    legendContainer.innerHTML = ''; // Clear existing

    data.forEach((item, index) => {
        const percentText = percentages[index].toFixed(2) + "%";
        const legendItem = document.createElement('div');
        legendItem.className = "flex items-center justify-between w-full mb-1";
        legendItem.innerHTML = `
            <div class="flex items-center">
                <span class="w-3 h-3 mr-2 rounded-full" style="background-color: ${bgColors[index]};"></span>
                ${item.country}
            </div>
            <span class="text-gray-400">${percentText}</span>
        `;
        legendContainer.appendChild(legendItem);
    });
}

// Initialize post time chart
function initializeReactionChart() {
    const postTimeCtx = document.getElementById('reactionChart')?.getContext('2d');
    if (postTimeCtx) {
        reactionChart = new Chart(postTimeCtx, {
            type: 'bar',
            data: {
                labels: [], // initially empty
                datasets: [{
                    label: 'Highest Sentiment (%)',
                    data: [],
                    backgroundColor: [],
                    borderColor: [],
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            color: '#9ca3af',
                            callback: function(value) {
                                return value + '%';
                            }
                        }, grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        title: {
                            display: true,
                            text: 'Percentage'
                        }
                    },
                    x: {
                        title: {
                            display: false,
                            text: 'Content Type'
                        },
                        ticks: {
                            color: '#9ca3af',
                            maxRotation: 45,
                            minRotation: 30
                        }
                    }
                },
                plugins: {
                    tooltips: {
                        callbacks: {
                            label: function(context) {
                                const sentiment = sentimentChart.data.datasets[0].sentimentTypes[context.dataIndex];
                                const value = Math.abs(context.raw).toFixed(2) + '%';
                                return `${sentiment}: ${value}`;
                            }
                        }
                    },
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
}

function updateReactionChart(data) {
    // Process data to get highest sentiment per content type
    const filteredData = data.filter(item => item.content_type !== "Music");

    // Process data to get highest sentiment per content type
    const processedData = filteredData.map(item => {
        const isPositiveHigher = item.positive_pct >= item.negative_pct;
        return {
            content_type: item.content_type,
            value: isPositiveHigher ? item.positive_pct * 100 : -item.negative_pct * 100, // Positive goes up, negative goes down
            sentiment: isPositiveHigher ? 'Positive' : 'Negative'
        };
    });

    const labels = processedData.map(d => d.content_type);
    const values = processedData.map(d => d.value.toFixed(2));
    const backgroundColors = processedData.map(d =>
        d.sentiment === 'Positive' ? 'rgba(34, 197, 94, 0.7)' : 'rgba(239, 68, 68, 0.7)'
    );
    const borderColors = backgroundColors.map(c => c.replace('0.7', '1'));
    const sentimentTypes = processedData.map(d => d.sentiment);

    // Update chart data
    reactionChart.data.labels = labels;
    reactionChart.data.datasets[0].data = values;
    reactionChart.data.datasets[0].backgroundColor = backgroundColors;
    reactionChart.data.datasets[0].borderColor = borderColors;
    reactionChart.data.datasets[0].sentimentTypes = sentimentTypes; // custom property for tooltip use

    reactionChart.update();
}

// Initialize comparison charts
function initializeComparisonCharts() {
    // Main Line Comparison Chart
    if (!comparisonChart) {
        const comparisonCtx = document.getElementById('comparisonChart')?.getContext('2d');
        if (comparisonCtx) {
            comparisonChart = new Chart(comparisonCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'IShowSpeed', data: [], borderColor: '#3b82f6', backgroundColor: 'transparent', tension: 0.4, borderWidth: 2 },
                        { label: 'Kai Cenat', data: [], borderColor: '#10b981', backgroundColor: 'transparent', tension: 0.4, borderWidth: 2 },
                        { label: 'Mr Beast', data: [], borderColor: '#f59e0b', backgroundColor: 'transparent', tension: 0.4, borderWidth: 2 },
                        { label: 'Pewdiepie', data: [], borderColor: '#8b5cf6', backgroundColor: 'transparent', tension: 0.4, borderWidth: 2 }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: { color: '#9ca3af', usePointStyle: true, pointStyle: 'circle' }
                        },
                        tooltip: {
                            callbacks: {
                                title: function (tooltipItems) {
                                    const label = tooltipItems[0].label;
                                    if (!label.includes('-')) return label;
                                    const [year, month] = label.split('-');
                                    const monthNames = ["January", "February", "March", "April", "May", "June",
                                        "July", "August", "September", "October", "November", "December"];
                                    return `${monthNames[parseInt(month, 10) - 1]} ${year}`;
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: {
                                color: '#9ca3af',
                                callback: value => formatNumber(value)
                            }
                        },
                        x: {
                            grid: { display: false },
                            ticks: {
                                color: '#9ca3af',
                                callback: function (value) {
                                    const label = this.getLabelForValue(value);
                                    const [year, month] = label.split("-");
                                    const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
                                    return monthNames[parseInt(month, 10) - 1];
                                }
                            }
                        }
                    }
                }
            });
        }
    }

    // Radar Chart - Engagement
    if (!engagementComparisonChart) {
        const ctx = document.getElementById('engagementComparisonChart')?.getContext('2d');
        if (ctx) {
            engagementComparisonChart = new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: ['Subscribers (M)', 'Total Views (B)', 'Video Count (100)', 'Engagement Per Video Rate (%)'],
                    datasets: [
                        { label: 'IShowSpeed', data: [], borderColor: '#3b82f6', backgroundColor: 'rgba(59, 130, 246, 0.2)', borderWidth: 2 },
                        { label: 'Kai Cenat', data: [], borderColor: '#10b981', backgroundColor: 'rgba(16, 185, 129, 0.2)', borderWidth: 2 },
                        { label: 'Mr Beast', data: [], borderColor: '#f59e0b', backgroundColor: 'rgba(245, 158, 11, 0.2)', borderWidth: 2 },
                        { label: 'Pewdiepie', data: [], borderColor: '#8b5cf6', backgroundColor: 'rgba(139, 92, 246, 0.2)', borderWidth: 2 }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: { color: '#9ca3af', usePointStyle: true, pointStyle: 'circle' }
                        }
                    },
                    scales: {
                        r: {
                            angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            pointLabels: { color: '#9ca3af' },
                            ticks: {
                                backdropColor: 'transparent',
                                color: '#9ca3af',
                                callback: value => formatNumber(value)
                            }
                        }
                    }
                }
            });
        }
    }

    // Doughnut Chart - Sentiment
    if (!sentimentComparisonChart) {
        const ctx = document.getElementById('sentimentComparisonChart')?.getContext('2d');
        if (ctx) {
            sentimentComparisonChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['IShowSpeed', 'Kai Cenat', 'Mr Beast', 'Pewdiepie'],
                    datasets: [{
                        data: [],
                        backgroundColor: ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: { color: '#9ca3af', padding: 10, usePointStyle: true, pointStyle: 'circle' }
                        }
                    },
                    cutout: '70%'
                }
            });
        }
    }

    // Bar Chart - Frequency
    if (!frequencyChart) {
        const ctx = document.getElementById('frequencyChart')?.getContext('2d');
        if (ctx) {
            frequencyChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['IShowSpeed', 'Kai Cenat', 'Mr Beast', 'Pewdiepie'],
                    datasets: [{
                        label: 'Videos per Month',
                        data: [],
                        backgroundColor: ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: '#9ca3af' }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { color: '#9ca3af' }
                        }
                    }
                }
            });
        }
    }
}


function updateComparisonCharts(data1, data2, data3, data4) {
    const map1 = Object.fromEntries(data1.map(d => [d.label, d.value]));
    const map2 = Object.fromEntries(data2.map(d => [d.label, d.value]));
    const map3 = Object.fromEntries(data3.map(d => [d.label, d.value]));
    const map4 = Object.fromEntries(data4.map(d => [d.label, d.value]));

    const labels1 = data1.map(d => d.label);
    const labels2 = data2.map(d => d.label);
    const labels3 = data3.map(d => d.label);
    const labels4 = data4.map(d => d.label);

    const commonLabels = labels1.filter(label => labels2.includes(label) && labels3.includes(label) && labels4.includes(label));

    const values1 = commonLabels.map(label => map1[label] ?? null);
    const values2 = commonLabels.map(label => map2[label] ?? null);
    const values3 = commonLabels.map(label => map3[label] ?? null);
    const values4 = commonLabels.map(label => map4[label] ?? null);

    if (comparisonChart) {
        comparisonChart.data.labels = commonLabels;
        comparisonChart.data.datasets[0].data = values1;
        comparisonChart.data.datasets[1].data = values2;
        comparisonChart.data.datasets[2].data = values3;
        comparisonChart.data.datasets[3].data = values4;
        comparisonChart.update();
    }
}

function updateEngagementComparisonCharts(data1, data2, data3, data4) {
    if (!engagementComparisonChart) return;

    const extractValues = (data) => [
        data.subscribers / 1_000_000,
        data.total_views / 1_000_000_000,
        data.video_count / 100,
        data["engagement_rate (%)"] * 100
    ];

    const dataset1 = extractValues(data1);
    const dataset2 = extractValues(data2);
    const dataset3 = extractValues(data3);
    const dataset4 = extractValues(data4);

    engagementComparisonChart.data.datasets[0].data = dataset1;
    engagementComparisonChart.data.datasets[1].data = dataset2;
    engagementComparisonChart.data.datasets[2].data = dataset3;
    engagementComparisonChart.data.datasets[3].data = dataset4;

    engagementComparisonChart.update();
}

function updateFrequencyComparisonCharts(data1, data2, data3, data4) {
    if (!frequencyChart) return;

    const videosPerMonth = [
        data1.videos_last_30_days || 0,
        data2.videos_last_30_days || 0,
        data3.videos_last_30_days || 0,
        data4.videos_last_30_days || 0
    ];

    frequencyChart.data.datasets[0].data = videosPerMonth;
    frequencyChart.update();
}

function updateSentimentComparisonCharts(data1, data2, data3, data4) {
    if (!sentimentComparisonChart) return;

    const computePositiveRatio = (data) => {
        const { positive, neutral, negative } = data.vader;
        const total = positive + neutral + negative;
        return total > 0 ? (positive / total) * 100 : 0;
    };

    const dataset = [
        computePositiveRatio(data1),
        computePositiveRatio(data2),
        computePositiveRatio(data3),
        computePositiveRatio(data4)
    ];

    sentimentComparisonChart.data.datasets[0].data = dataset;
    sentimentComparisonChart.update();
}

function updateOpportunities(data) {
    const container = document.getElementById("contentRecommendations");
    if (!container || !data || !data.response) return;

    let cleanedContent = data.response;

    // Optional: remove markdown bold markers
    cleanedContent = cleanedContent.replace(/\*\*(.*?)\*\*/g, '$1');

    // Optional: convert list dashes to <ul><li>
    cleanedContent = cleanedContent.replace(/^- (.*)$/gm, '<li>$1</li>');
    cleanedContent = cleanedContent.replace(/(<li>[\s\S]*?<\/li>)/g, '<ul class="list-disc pl-5 text-sm text-gray-300">$1</ul>');

    // Optional: convert headings
    cleanedContent = cleanedContent.replace(/^# (.*)$/gm, '<h2 class="text-lg font-bold mt-4 mb-2">$1</h2>');
    cleanedContent = cleanedContent.replace(/^## (.*)$/gm, '<h3 class="text-md font-semibold mt-3 mb-1">$1</h3>');
    cleanedContent = cleanedContent.replace(/^### (.*)$/gm, '<h4 class="text-md font-semibold mt-3 mb-1">$1</h4>');

    // Wrap remaining lines as paragraphs
    cleanedContent = cleanedContent.replace(/^(?!<h|<ul|<li)(.+)$/gm, '<p class="mb-2">$1</p>');

    container.innerHTML = cleanedContent;
}



// Helper function to format numbers
function formatNumber(num) {
    if (num >= 1_000_000_000) {
        return (num / 1_000_000_000).toFixed(1) + 'B';
    } else if (num >= 1_000_000) {
        return (num / 1_000_000).toFixed(1) + 'M';
    } else if (num >= 1_000) {
        return (num / 1_000).toFixed(1) + 'K';
    } else {
        return num.toString();
    }
}
