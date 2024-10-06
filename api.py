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


def ACTION_SET_TIMER(duration: str, EXTRA_MESSAGE: str="", EXTRA_SKIP_UI: bool=True) -> None:
    """
    Set a countdown timer with the given parameters.
    
    Args:
        duration (str): The duration of the timer in the format "HH hours MM minutes SS seconds".
            For example, "1 hours 30 minutes" or "10 minutes" or "1 hours 30 minutes 15 seconds", etc.
        
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


def ACTION_INSERT_EVENT(TITLE: str, DESCRIPTION: str, EVENT_LOCATION: str=None, 
                        EXTRA_EVENT_ALL_DAY: bool=False, 
                        EXTRA_EVENT_BEGIN_TIME: str=None, 
                        EXTRA_EVENT_END_TIME: str=None, 
                        EXTRA_EMAIL: List[str]=None) -> None:
    """
    Add a new event to the user's calendar.
    
    Args:
        TITLE (str): The event title.
        
        DESCRIPTION (str): The event description.
        
        EVENT_LOCATION (str): The event location. Default is None.
        
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

def ACTION_PICK(data_type: Literal["ALL", "PHONE", "EMAIL", "ADDRESS"] = "ALL") -> str:
    """
    This function allows the user to select a contact or specific contact information (such as phone
    number, email, or postal address) and returns a content URI for the selected data.

    Args:
        data_type (str): The type of contact data to pick. Default is "ALL".
            Available options:
            - "ADDRESS": Pick a contact's address
            - "PHONE": Pick a contact's phone number
            - "EMAIL": Pick a contact's email address
            - "ALL": Pick the entire contact

    Returns:
        str: A content URI as a string, pointing to the selected contact or contact data.
            This URI can be used to query for more details about the contact.
    """
    pass  # Placeholder for actual implementation

def get_contact_info(name: str, key: str)->str:
    """
    Get the contact information based on the contact name and the key.

    Args:
        name (str): The name of the contact.
        key (str): The key to get the information of the contact.
            can be one of the following: "email", "phone", "address" "uri"
            if key is "uri", this function will return the uri of the contact that can be 
            used to edit the contact.

    Returns:
        str: The information of the contact based on the key.
        
    Example:
        get_contact_info("John Doe", "email")
        this will return the email of the contact named "John Doe"
    """
    pass  # Placeholder for actual implementation

def get_contact_info_from_uri(contact_uri: str, key: str)->str:
    """
    Get the contact information based on the contact URI and the key.

    Args:
        contact_uri (str): The URI of the contact.
        key (str): The key to get the information of the contact.
            can be one of the following: "email", "phone", "address"

    Returns:
        str: The information of the contact based on the key.
        
    Example:
        get_contact_info_from_uri("content://com.android.contacts/data/9", "email")
        this will return the email of the contact with URI "content://com.android.contacts/data/9"
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
    Note:
        The contact_uri can be obtained in two primary ways:
        1. Using the contact URI returned by the ACTION_PICK function.
        2. Accessing the list of all contacts directly (requires appropriate permissions).

    Args:
        contact_uri (str): A content URI as a string, pointing to the contact
            to be edited. This URI can be obtained from the ACTION_PICK function
            or by querying the contacts database.
        contact_info (Optional[Dict[str, Any]]): A dictionary containing additional
            contact information to pre-fill in the edit form. Keys should correspond
            to contact fields (available key: 'email', 'phone', 'name', 'company', 'address'), and values should be
            the data to pre-fill. Default is None.

    Returns:
        None
    """
    pass  # Placeholder for actual implementation


from typing import Dict, Any

def ACTION_INSERT_CONTACT(contact_info: Dict[str, Any]) -> None:
    """
    Insert a new contact.

    This function allows the user to create a new contact with the provided
    contact information. It will open the contact creation interface with
    pre-filled information based on the provided data.

    Args:
        contact_info (Dict[str, Any]): A dictionary containing the contact
            information to pre-fill in the new contact form. Keys should
            correspond to contact fields (available key: 'email', 'phone', 'name', 'company', 'address'),
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


from typing import List, Optional, Union
from pathlib import Path

from typing import List, Optional, Union

def send_email(
    to: List[str],
    subject: str,
    body: str,
    cc: Optional[List[str]] = None,
    bcc: Optional[List[str]] = None,
    attachments:  List[str] = None
) -> None:
    """
    Compose and send an email with optional attachments.

    This function allows the user to compose an email with various options,
    including multiple recipients, CC, BCC, and file attachments.

    Args:
        to (List[str]): A list of recipient email addresses.
        subject (str): The subject of the email.
        body (str): The body text of the email.
        cc (Optional[List[str]]): A list of CC recipient email addresses. Default is None.
        bcc (Optional[List[str]]): A list of BCC recipient email addresses. Default is None.
        attachments (List[str]): list of URIs
            pointing to the files to be attached to the email. These can be file URIs,
            content URIs, or any other valid Android resource URI. Default is None (meaning no attachments). 

    Returns:
        None

    Examples:
        # Send an email with a content URI attachment
        send_email(
            to=["recipient@example.com"],
            subject="Document",
            body="Please find the attached document.",
            attachments=["content://com.android.providers.downloads.documents/document/1234"]
        )

        # Send an email with multiple attachments using different URI types
        send_email(
            to=["team@example.com"],
            subject="Project Files",
            body="Here are the latest project files.",
            attachments=[
                "content://media/external/images/media/5678",
                "content://com.android.externalstorage.documents/document/primary%3ADownload%2Freport.pdf"
            ]
        )
    """
    pass  # Placeholder for actual implementation

def send_message(phone_number: str, subject: str, body: str, attachments: List[str]=None) -> None:
    """
    Send a message with attachments.
    
    This function helps user to compose and send a message with optional attachments to a phone number.

    Args:
        phone_number (str): The phone number to send the message to.
        subject (str): The subject of the message.
        body (str): The body text of the message.
        attachments (List[str]): A list of URIs pointing to the files to be attached to the message.
            Default is None (meaning no attachments).

    Returns:
        None
    """
    pass


from typing import List, Optional

def ACTION_GET_RINGTONE() -> Optional[str]:
    """
    Let user select a ringtone and return the URI of the selected ringtone.
    
    This function allows the user to select a ringtone from the device's ringtone picker.
    It returns the content URI of the selected ringtone that can be use to set alarm.
        
        
    Returns:
        Optional[str]: A content URI as a string pointing to the selected ringtone.
            If no ringtone is selected or the operation is cancelled, returns None.
    """
    
    return None

def ACTION_GET_CONTENT(
    mime_type: str,
    allow_multiple: bool = False,
) -> List[str]:
    """
    Let user select one or multilple file(s) of a specific type.

    This function allows the user to select one or more files of a specified MIME type.
    It returns a list of content URIs for the selected file(s).

    Args:
        mime_type (str): The MIME type of the file(s) to be selected (e.g., "image/*", "audio/*", "video/*", "*/*").
        allow_multiple (bool): If True, allows selection of multiple files. Defaults to False.

    Returns:
        List[str]: A list of URIs as strings, each pointing to a selected file.
                   If no file is selected or the operation is cancelled, returns an empty list.


    Examples:
        # Select a single image
        image_uris = ACTION_GET_CONTENT("image/*")

        # Select multiple documents
        doc_uris = ACTION_GET_CONTENT("application/pdf", allow_multiple=True)
    """
    # Placeholder implementation
    # In a real Android app, this would launch an intent and handle the result
    # Here, we'll just return an empty list
    return []


from typing import List, Union, Optional

def ACTION_OPEN_DOCUMENT(
    mime_types: List[str],
    allow_multiple: bool = False,
) -> List[str]:
    """
    Opens a file or multiple files of specified MIME type(s).

    This function allows the user to select one or more files of specified MIME type(s).
    It provides long-term, persistent access to the selected file(s). This is usually better than using ACTION_GET_CONTENT, since it can also access files from cloud storage or other document providers.

    Args:
        mime_types (List[str]): The MIME type(s) of the file(s) to be selected.
            Can be a list of strings for multiple types or only a list with a single string for a single type.
        allow_multiple (bool, optional): If True, allows selection of multiple files. Defaults to False.

    Returns:
        List[str]: A list of content URIs as strings, each pointing to a selected file.
                   If no file is selected or the operation is cancelled, returns an empty list.

    Examples:
        # Open a single image
        image_uris = ACTION_OPEN_DOCUMENT(["image/*"])

        # Open multiple documents of different types
        doc_uris = ACTION_OPEN_DOCUMENT(["application/pdf", "text/plain"], allow_multiple=True)
    """
    # Placeholder implementation
    # In a real environment, this would open a file picker and return the selected file(s)
    return []

from typing import Optional

def ACTION_CREATE_DOCUMENT(
    mime_type: str,
    initial_name: str,
) -> Optional[str]:
    """
    Creates a new document that app can write to. And user can select where they'd like to create it.

    Instead of selecting from existing PDF documents, 
    the ACTION_CREATE_DOCUMENT lets users select where they'd like to create a new document, such as within another app that manages the document's storage. 
    And then return the URI location of document that you can read from and write to.

    Args:
        mime_type (str): The MIME type of the document to be created (e.g., "text/plain", "application/pdf").
        initial_name (str): The suggested name for the new document.

    Returns:
        Optional[str]: A URI as a string pointing to the newly created document.
                       Returns None if the operation is cancelled or fails.
                       
    Examples:
        # Create a new text document
        new_doc_uri = ACTION_CREATE_DOCUMENT("text/plain", "New Document.txt")

        # Create a new PDF file
        new_pdf_uri = ACTION_CREATE_DOCUMENT("application/pdf", "Report.pdf")

        # Create a new image file
        new_image_uri = ACTION_CREATE_DOCUMENT("image/jpeg", "Photo.jpg")
    """
    # Placeholder implementation
    # In a real environment, this would open a file creation dialog and return the URI of the new file
    return None


def search_location(query: str)->None:
    """
    Search for a location using a query string in a map application for user.

    Args:
        query (str): The search query string to find a location.
    """
    pass  # Placeholder for actual implementation


def dial(phone_number: str) -> None:
    """
    Opens the dialer with a specified number in a phone app for user.

    This function helps user to start a phone call process. It can open
    the dialer with a pre-filled number. User can then choose to dial the number.

    Args:
        phone_number (str): The phone number to dial. This should be a valid
            telephone number as defined in IETF RFC 3966. Examples include:
            "2125551212" or "(212) 555 1212".

    Examples:
        # Open dialer with a number
        dial("2125551212")
    """
    # Function implementation goes here
    pass

from typing import Optional

def web_search(query: str, engine:str="baidu") -> None:
    """
    Initiates a web search using the specified query.

    This function starts a web search using the default search engine.
    It opens the search results in the default web browser or appropriate search application.

    Args:
        query (str): The search string or keywords to be used for the web search.
        engine (str): The search engine to use. Default is "baidu".
            Possible values are: "baidu", "google"
        
    Examples:
        # Perform a simple web search
        web_search("Python programming tutorials")

        # Search for a phrase
        web_search('"to be or not to be"')
        
        # Search using a specific search engine
        web_search("Python programming tutorials", "google")

    """
    # Function implementation goes here
    pass

def open_settings(setting_type: str = "general") -> None:
    """
    Opens a specific settings screen on the device.

    This function allows you to open various system settings screens,
    providing quick access to different device configuration options.

    Args:
        setting_type (str): The type of settings screen to open.
            Possible values are:
            - "general": General settings (default)
            - "wireless": Wireless & network settings
            - "airplane_mode": Airplane mode settings
            - "wifi": Wi-Fi settings
            - "apn": APN settings
            - "bluetooth": Bluetooth settings
            - "date": Date & time settings
            - "locale": Language & input settings
            - "input_method": Input method settings
            - "display": Display settings
            - "security": Security settings
            - "location": Location settings
            - "internal_storage": Internal storage settings
            - "memory_card": Memory card settings

    Examples:
        # Open general settings
        open_settings()

        # Open Wi-Fi settings
        open_settings("wifi")

        # Open Bluetooth settings
        open_settings("bluetooth")
    """
    # Function implementation goes here
    pass
