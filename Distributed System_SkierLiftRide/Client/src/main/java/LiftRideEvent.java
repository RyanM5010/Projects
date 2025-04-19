import io.swagger.client.model.LiftRide;

public class LiftRideEvent {
  private final int resortID;
  private final int skierID;
  private final String seasonID;
  private final String dayID;
  private final LiftRide liftRide;

  public LiftRideEvent (int resortID, int skierID, String seasonID, String dayID, LiftRide liftRide) {
    this.resortID = resortID;
    this.skierID = skierID;
    this.seasonID = seasonID;
    this.dayID = dayID;
    this.liftRide = liftRide;
  }

  public int getResortID() {
    return resortID;
  }

  public int getSkierID() {
    return skierID;
  }

  public String getSeasonID() {
    return seasonID;
  }

  public String getDayID() {
    return dayID;
  }

  public LiftRide getLiftRide() {
    return liftRide;
  }
}
