package commons.shared.model;

public class LiftRideMessage {
  private int resortId;
  private int seasonId;
  private int dayId;
  private int skierId;
  private LiftRide liftRide;

  public LiftRideMessage() {
}

  public LiftRideMessage(int resortId, int seasonId, int dayId, int skierId, LiftRide liftRide) {
    this.resortId = resortId;
    this.seasonId = seasonId;
    this.dayId = dayId;
    this.skierId = skierId;
    this.liftRide = liftRide;
  }

  public int getResortId() {
    return resortId;
  }

  public void setResortId(int resortId) {
    this.resortId = resortId;
  }

  public int getSeasonId() {
    return seasonId;
  }

  public void setSeasonId(int seasonId) {
    this.seasonId = seasonId;
  }

  public int getDayId() {
    return dayId;
  }

  public void setDayId(int dayId) {
    this.dayId = dayId;
  }

  public int getSkierId() {
    return skierId;
  }

  public void setSkierId(int skierId) {
    this.skierId = skierId;
  }

  public LiftRide getLiftRide() {
    return liftRide;
  }

  public void setLiftRide(LiftRide liftRide) {
    this.liftRide = liftRide;
  }
}
