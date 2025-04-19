import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;

public class EventGenerator implements Runnable {
  private final BlockingQueue<LiftRideEvent> eventQueue;
  private final int totalEvents;
  private final String dayId;
  private volatile boolean running = true;
  private static final int QUEUE_TIME_OUT = 50;

  public EventGenerator(BlockingQueue<LiftRideEvent> eventQueue, int totalEvents, int dayId) {
    this.eventQueue = eventQueue;
    this.totalEvents = totalEvents;
    this.dayId = String.valueOf(dayId); // Convert to String to match API
  }

  @Override
  public void run() {
    try {
      for (int i = 0; i < totalEvents && running; i++) {
        LiftRideEvent event = DataGenerator.generateLiftRide(dayId);

        while (!eventQueue.offer(event, QUEUE_TIME_OUT, TimeUnit.MILLISECONDS)) {
          if (!running) return;
          Thread.sleep(10); // Brief pause before retrying
        }
      }
      System.out.println("Event Generator completed: " + totalEvents + " events generated");
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      System.err.println("Event Generator interrupted");
    }
  }

  public void shutdown() {
    running = false;
  }
}
