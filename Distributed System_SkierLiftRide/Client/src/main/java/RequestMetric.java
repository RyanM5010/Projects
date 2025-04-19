public class RequestMetric {
    private final long startTime;
    private final String requestType;
    private final long latency;
    private final int responseCode;

    public RequestMetric(long startTime, String requestType, long latency, int responseCode) {
        this.startTime = startTime;
        this.requestType = requestType;
        this.latency = latency;
        this.responseCode = responseCode;
    }

    public String toCsvString() {
        return String.format("%d,%s,%d,%d", startTime, requestType, latency, responseCode);
    }

    public long getStartTime() { return startTime; }
    public String getRequestType() { return requestType; }
    public long getLatency() { return latency; }
    public int getResponseCode() { return responseCode; }
} 