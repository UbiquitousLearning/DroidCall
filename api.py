from typing import List, Dict, Any

def ACTION_SET_ALARM(EXTRA_HOUR: int, EXTRA_MINUTES: int, EXTRA_MESSAGE: str="", 
                     EXTRA_DAYS: list[str]=None, EXTRA_RINGTONE: str=None,
                     EXTRA_VIBRATE: bool=False, EXTRA_SKIP_UI: bool=True) -> None:
    """
    Set an alarm with the given parameters.
    
    Args:
        EXTRA_HOUR (int): The hour of the alarm in 24-hour format.
        
        EXTRA_MINUTES (int): The minutes of the alarm.
        
        EXTRA_MESSAGE (str): The message of the alarm. Default is an empty string.
        
        EXTRA_DAYS (list[str]): The days of the alarm, e.g. ["Monday", "Tuesday"]. Default is None.
        
        EXTRA_RINGTONE (str): The ringtone of the alarm specified by a content URI. Default is None.
            if None, the default ringtone will be used. If set to "silent", no ringtone will be played.
            
        EXTRA_VIBRATE (bool): Whether the alarm should vibrate. Default is False.
        
        EXTRA_SKIP_UI (bool): A boolean specifying whether the responding app must skip its UI when setting the alarm.
            If true, the app must bypass any confirmation UI and set the specified alarm. Default is True.
    """
    pass


def ACTION_SET_TIMER(EXTRA_LENGTH: int, EXTRA_MESSAGE: str="", EXTRA_SKIP_UI: bool=True) -> None:
    """
    Set a countdown timer with the given parameters.
    
    Args:
        EXTRA_LENGTH (int): The length of the timer in seconds.
        
        EXTRA_MESSAGE (str): A custom message to identify the timer. Default is an empty string.
        
        EXTRA_SKIP_UI (bool): A boolean specifying whether the responding app must skip its UI when setting the timer.
            If true, the app must bypass any confirmation UI and start the specified timer. Default is True.
    """
    pass


def ACTION_SHOW_ALARMS() -> None:
    """
    Show the list of current alarms.
    """
    pass


def ACTION_INSERT_EVENT(TITLE: str, DESCRIPTION: str, EVENT_LOCATION: str, 
                        EXTRA_EVENT_ALL_DAY: bool=False, 
                        EXTRA_EVENT_BEGIN_TIME: str=None, 
                        EXTRA_EVENT_END_TIME: str=None, 
                        EXTRA_EMAIL: List[str]=None) -> None:
    """
    Add a new event to the user's calendar.
    
    Args:
        TITLE (str): The event title.
        
        DESCRIPTION (str): The event description.
        
        EVENT_LOCATION (str): The event location.
        
        EXTRA_EVENT_ALL_DAY (bool): A boolean specifying whether this is an all-day event. Default is False.
        
        EXTRA_EVENT_BEGIN_TIME (str): The start time of the event in ISO 8601 format. Default is None.
        
        EXTRA_EVENT_END_TIME (str): The end time of the event in ISO 8601 format. Default is None.
        
        EXTRA_EMAIL (List[str]): A list of email addresses that specify the invitees. Default is None.
    """
    pass


def ACTION_IMAGE_CAPTURE() -> str:
    """
    Capture a picture using the camera app and return the URI of the saved photo.
    
    This function uses the ACTION_IMAGE_CAPTURE intent to open the camera app and capture a photo.
    The photo is saved to a URI location, which is returned by this function.
    User can then use this URI to access the photo file and do whatever they want with it.
    
    Returns:
        str: The URI location where the camera app saves the photo file.
    """
    pass


def ACTION_VIDEO_CAPTURE() -> str:
    """
    Capture a video using the camera app and return the URI of the saved video.
    
    This function uses the ACTION_VIDEO_CAPTURE intent to open the camera app and capture a video.
    The video is saved to a URI location, which is returned by this function.
    User can then use this URI to access the video file and do whatever they want with it.
    
    Returns:
        str: The URI location where the camera app saves the video file.
    """
    # Here, we simulate generating a URI for the captured video.
    pass


def INTENT_ACTION_STILL_IMAGE_CAMERA() -> None:
    """
    Open a camera app in still image mode for capturing photos for user.
    """
    pass


def INTENT_ACTION_VIDEO_CAMERA() -> None:
    """
    Open a camera app in video mode to start recording a video.
    """
    pass


from typing import Literal

def ACTION_PICK(data_type: Literal["ALL", "PHONE", "EMAIL", "POSTAL"] = "ALL") -> str:
    """
    This function allows the user to select a contact or specific contact information (such as phone
    number, email, or postal address) and returns a content URI for the selected data.

    Args:
        data_type (str): The type of contact data to pick. Default is "ALL".
            Available options:
            - "ALL": Pick a full contact
            - "PHONE": Pick a contact's phone number
            - "EMAIL": Pick a contact's email address
            - "POSTAL": Pick a contact's postal address

    Returns:
        str: A content URI as a string, pointing to the selected contact or contact data.
            This URI can be used to query for more details about the contact.
    """
    pass  # Placeholder for actual implementation


def ACTION_VIEW_CONTACT(contact_uri: str) -> None:
    """
    Display the details for a known contact.

    This function allows the user to view the details of a specific contact
    based on the provided contact URI.

    Args:
        contact_uri (str): A content URI as a string, pointing to the contact
            whose details should be displayed. This URI can be obtained from
            the ACTION_PICK function or by querying the contacts database.

    Returns:
        None
    """
    pass  # Placeholder for actual implementation


from typing import Optional, Dict, Any

def ACTION_EDIT_CONTACT(contact_uri: str, contact_info: Optional[Dict[str, Any]] = None) -> None:
    """
    Edit an existing contact.

    This function allows the user to edit the details of a specific contact
    based on the provided contact URI. Additional contact information can be
    provided to pre-fill certain fields in the edit form.

    Args:
        contact_uri (str): A content URI as a string, pointing to the contact
            to be edited. This URI can be obtained from the ACTION_PICK function
            or by querying the contacts database.
        contact_info (Optional[Dict[str, Any]]): A dictionary containing additional
            contact information to pre-fill in the edit form. Keys should correspond
            to contact fields (e.g., 'email', 'phone', 'name'), and values should be
            the data to pre-fill. Default is None.

    Returns:
        None

    Note:
        The contact_uri can be obtained in two primary ways:
        1. Using the contact URI returned by the ACTION_PICK function.
        2. Accessing the list of all contacts directly (requires appropriate permissions).
    """
    pass  # Placeholder for actual implementation


from typing import Dict, Any

def ACTION_INSERT(contact_info: Dict[str, Any]) -> None:
    """
    Insert a new contact.

    This function allows the user to create a new contact with the provided
    contact information. It will open the contact creation interface with
    pre-filled information based on the provided data.

    Args:
        contact_info (Dict[str, Any]): A dictionary containing the contact
            information to pre-fill in the new contact form. Keys should
            correspond to contact fields (e.g., 'name', 'email', 'phone'),
            and values should be the data to pre-fill.

    Returns:
        None

    Example:
        ACTION_INSERT({
            "name": "John Doe",
            "email": "john.doe@example.com",
            "phone": "1234567890"
        })
    """
    pass  # Placeholder for actual implementation

