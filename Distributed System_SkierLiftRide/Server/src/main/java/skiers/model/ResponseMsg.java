package skiers.model;

public class ResponseMsg {
  private String message;

  // Add default constructor
  public ResponseMsg() {
  }

  // Add constructor that takes a message
  public ResponseMsg(String message) {
    this.message = message;
  }

  public String getMessage() {
    return message;
  }

  public void setMessage(String message) {
    this.message = message;
  }
}
